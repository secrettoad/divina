import boto3
import json
import os
import datetime
import io
from botocore.exceptions import ClientError
import backoff
import subprocess
from pkg_resources import resource_filename
import math


####TODO abtract rootish from role jsons - use os.path.expandvars
####TODO abstract bucket names
####TODO setup EC2 SSH keypair when creating cluster


def create_source_s3_role(boto3_session, iam_path='AWS_IAM'):
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), iam_path, 'S3_SOURCE_IAM')) as f:
        policy2_json = os.path.expandvars(json.dumps(json.load(f)))
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), iam_path, 'S3_SOURCE_ROLE_TRUST_POLICY')) as f:
        s3_source_role_trust_policy = os.path.expandvars(json.dumps(json.load(f)))

    source_iam = boto3_session.client('iam')

    try:
        role = source_iam.create_role(
            Path='/',
            RoleName='s3SourceRole' + os.environ['VISION_ID'],
            AssumeRolePolicyDocument=s3_source_role_trust_policy,
            Description='role for accessing data in source s3'
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print(e)

    # creates role in source account that has s3 permissions
    policy_step2 = source_iam.create_policy(
        PolicyName='s3SourceRole' + os.environ['VISION_ID'],
        PolicyDocument=policy2_json
    )
    source_iam.attach_role_policy(
        RoleName='s3SourceRole' + os.environ['VISION_ID'], PolicyArn=policy_step2['Policy']['Arn'])

    return role


def vision_setup(source_session, source_bucket, worker_profile, executor_role, vision_session, region):
    source_s3 = source_session.client('s3')
    vision_s3 = vision_session.client('s3')

    try:
        vision_s3.create_bucket(Bucket='coysu-divina-prototype-visions',
                                CreateBucketConfiguration={
                                    'LocationConstraint': region
                                })
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print(e)
        else:
            raise e
    try:
        source_objects = source_s3.list_objects(Bucket=source_bucket)['Contents']
    except KeyError as e:
        raise e

    for file in source_objects:
        object_response = source_s3.get_object(
            Bucket=source_bucket,
            Key=file['Key']
        )
        vision_s3.put_object(
            Body=io.BytesIO(object_response['Body'].read()),
            Bucket='coysu-divina-prototype-visions',
            Key='coysu-divina-prototype-{}/data/{}'.format(os.environ['VISION_ID'], file['Key'])
        )
        create_ec2_partitioning_instance(vision_session,
                                         key='coysu-divina-prototype-{}/data/{}'.format(os.environ['VISION_ID'],
                                                                                        file['Key']),
                                         bucket='coysu-divina-prototype-visions')

    spark_scripts_directory = os.path.join(os.path.abspath('.'), 'spark_scripts')

    for file in os.listdir(spark_scripts_directory):
        path = os.path.join(spark_scripts_directory, file)
        vision_s3.upload_file(
            Filename=path,
            Bucket='coysu-divina-prototype-visions',
            Key='coysu-divina-prototype-{}/spark_scripts/{}'.format(os.environ['VISION_ID'], file)
        )

    emr_cluster = setup_emr_cluster(vision_session=vision_session, model_file='spark_model.py',
                                    worker_profile=worker_profile, executor_role=executor_role)


def create_vision_iam_role(boto3_session, iam_path='AWS_IAM'):
    vision_iam = boto3_session.client('iam')

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), iam_path, 'DIVINA_IAM')) as f:
        policy1_json = os.path.expandvars(json.dumps(json.load(f)))
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), iam_path, 'DIVINA_ROLE_TRUST_POLICY')) as f:
        vision_role_trust_policy = os.path.expandvars(json.dumps(json.load(f)))

    try:
        role = vision_iam.create_role(
            Path='/',
            RoleName='visionRole' + os.environ['VISION_ID'],
            AssumeRolePolicyDocument=vision_role_trust_policy,
            Description='role for populating step2 s3 bucket from step1'
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            print(e)
        else:
            raise e

    policy_step1 = vision_iam.create_policy(
        PolicyName='visionRole' + os.environ['VISION_ID'],
        PolicyDocument=policy1_json
    )

    vision_iam.attach_role_policy(
        RoleName='visionRole' + os.environ['VISION_ID'], PolicyArn=policy_step1['Policy']['Arn'])

    return role


def create_emr_roles(boto3_session):
    iam_client = boto3_session.client('iam')

    if not all(x in [r['RoleName'] for r in iam_client.list_roles()['Roles']] for x in
               ['EMR_EC2_DefaultRole', 'EMR_DefaultRole']):
        subprocess.run(['aws', 'emr', 'create-default-roles'])

    return iam_client.list_roles()


def delete_emr_cluster(emr_cluster, boto3_session):
    emr_client = boto3_session.client('emr')
    emr_client.terminate_job_flows(JobFlowIds=[emr_cluster['JobFlowId']])


def setup_emr_cluster(vision_session, model_file, worker_profile='EMR_EC2_DefaultRole',
                      executor_role='EMR_DefaultRole'):
    emr_client = vision_session.client('emr')
    cluster = emr_client.run_job_flow(
        Name="divina-cluster-{}".format(os.environ['VISION_ID']),
        ReleaseLabel='emr-5.12.0',
        Instances={
            'KeepJobFlowAliveWhenNoSteps': False,
            'TerminationProtected': False,
            'InstanceGroups': [
                {
                    'InstanceRole': 'TASK',
                    'InstanceType': 'c4.xlarge',
                    'InstanceCount': 3,
                    'Configurations': [{
                        "Classification": "spark-env",
                        "Configurations": [
                            {
                                "Classification": "export",
                                "Properties": {
                                    "PYSPARK_PYTHON": "/usr/bin/python3",
                                    "VISION_ID": os.environ['VISION_ID']
                                }
                            }
                        ]
                    }]
                },
                {
                    'InstanceRole': 'MASTER',
                    'InstanceType': 'c4.xlarge',
                    'InstanceCount': 1,
                    'Configurations': [{
                        "Classification": "spark-env",
                        "Configurations": [
                            {
                                "Classification": "export",
                                "Properties": {
                                    "PYSPARK_PYTHON": "/usr/bin/python3",
                                    "VISION_ID": os.environ['VISION_ID']
                                }
                            }
                        ]
                    }]
                },
                {
                    'InstanceRole': 'CORE',
                    'InstanceType': 'c4.xlarge',
                    'InstanceCount': 1,
                    'Configurations': [{
                        "Classification": "spark-env",
                        "Configurations": [
                            {
                                "Classification": "export",
                                "Properties": {
                                    "PYSPARK_PYTHON": "/usr/bin/python3",
                                    "VISION_ID": os.environ['VISION_ID']
                                }
                            }
                        ]
                    }]
                }
            ]

        },
        LogUri='s3://coysu-divina-prototype-{}/emr_logs'.format(os.environ['VISION_ID']),
        Applications=[
            {
                'Name': 'Spark'
            }
        ],

        Steps=[
            {
                'Name': 'Setup Debugging',
                'ActionOnFailure': 'TERMINATE_CLUSTER',
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': ['state-pusher-script']
                }
            },
            {
                'Name': 'setup - copy files',
                'ActionOnFailure': 'CANCEL_AND_WAIT',
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': ['aws', 's3', 'sync',
                             's3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/spark_scripts'.format(
                                 os.environ['VISION_ID']),
                             '/home/hadoop/']
                }
            },
            {
                'Name': 'Run Spark',
                'ActionOnFailure': 'CANCEL_AND_WAIT',
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': ['spark-submit', '/home/hadoop/{}'.format(model_file)]
                }
            }
        ],
        Configurations=[{
            "Classification": "spark-env",
            "Configurations": [
                {
                    "Classification": "export",
                    "Properties": {
                        "PYSPARK_PYTHON": "/usr/bin/python3"
                    }
                }
            ]
        }],
        VisibleToAllUsers=True,
        JobFlowRole=worker_profile,
        ServiceRole=executor_role
    )
    return cluster


def ec2_pricing(boto3_session, filter_params=None):
    pricing = boto3_session.client('pricing', region_name='us-east-1')
    products_params = {'ServiceCode': 'AmazonEC2',
                       'Filters': [
                           {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
                           {'Type': 'TERM_MATCH', 'Field': 'location',
                            'Value': '{}'.format(get_region_name(boto3_session.region_name))}
                       ]}
    if filter_params:
        products_params['Filters'] = products_params['Filters'] + filter_params
    while True:
        response = pricing.get_products(**products_params)
        yield from [i for i in response['PriceList']]
        if 'NextToken' not in response:
            break
        products_params['NextToken'] = response['NextToken']
    return response


def unnest_ec2_price(product):
    od = product['terms']['OnDemand']
    id1 = list(od)[0]
    id2 = list(od[id1]['priceDimensions'])[0]
    return {od[id1]['priceDimensions'][id2]['unit'] + '_USD': od[id1]['priceDimensions'][id2]['pricePerUnit']['USD']}


def get_region_name(region_code):
    default_region = 'EU (Ireland)'
    endpoint_file = resource_filename('botocore', 'data/endpoints.json')
    try:
        with open(endpoint_file, 'r') as f:
            data = json.load(f)
        return data['partitions'][0]['regions'][region_code]['description']
    except IOError:
        return default_region


def create_ec2_partitioning_instance(vision_session, key, bucket):
    size = get_s3_object_size(vision_session, key, bucket)
    required_gb = math.ceil(size * 2 / 1000000)
    instance_info = [json.loads(p) for p in ec2_pricing(vision_session) if
                     'memory' in json.loads(p)['product']['attributes'] and 'OnDemand' in json.loads(p)['terms']]
    available_instance_types = [dict(i, **unnest_ec2_price(i)) for i in instance_info if
                                i['product']['attributes']['memory'].split(' ')[0].isdigit()]
    eligible_instance_types = [i for i in available_instance_types if
                               float(i['product']['attributes']['memory'].split(' ')[0]) >= required_gb and 'Hrs_USD' in i]
    partitioning_instance_type = eligible_instance_types[min(range(len(eligible_instance_types)), key=lambda index:
    eligible_instance_types[index]['Hrs_USD'])]


    ###TODO START HERE FILTER OUT $0 DEDICATED INSTANCE TYPES




    ec2 = vision_session.client('ec2')
    ec2.create_instances(ImageId='<ami-image-id>', MinCount=1, MaxCount=5,
                         InstanceType=partitioning_instance_type['product']['attributes']['instanceType'], CpuOptions={
            'CoreCount': 123,
            'ThreadsPerCore': 123
        })


def get_s3_object_size(vision_session, key, bucket):
    s3 = vision_session.client('s3')
    response = s3.head_object(Bucket=bucket, Key=key)
    size = response['ContentLength']
    return size


@backoff.on_exception(backoff.expo,
                      ClientError)
def assume_role(sts_client, role_arn, session_name):
    assumed_import_role = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName=session_name
    )
    return assumed_import_role


def create_vision(source_role=None, vision_role=None, worker_profile='EMR_EC2_DefaultRole',
                  executor_role='EMR_DefaultRole', region='us-east-2'):
    os.environ['VISION_ID'] = str(round(datetime.datetime.now().timestamp()))

    source_session = boto3.session.Session(aws_access_key_id=os.environ['SOURCE_AWS_PUBLIC_KEY'],
                                           aws_secret_access_key=os.environ['SOURCE_AWS_SECRET_KEY'],
                                           region_name=region)

    vision_session = boto3.session.Session(aws_access_key_id=os.environ['AWS_PUBLIC_KEY'],
                                           aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name=region)

    if not vision_role:
        vision_role = create_vision_iam_role(boto3_session=vision_session)

        vision_sts_client = vision_session.client('sts', aws_access_key_id=os.environ['SOURCE_AWS_PUBLIC_KEY'],
                                                  aws_secret_access_key=os.environ['SOURCE_AWS_SECRET_KEY'])

        assumed_vision_role = assume_role(sts_client=vision_sts_client,
                                          role_arn="arn:aws:iam::{}:role/{}".format(
                                              os.environ['ACCOUNT_NUMBER'], vision_role['Role']['RoleName']),
                                          session_name="AssumeRoleSession2")

        # From the response that contains the assumed role, get the temporary
        # credentials that can be used to make subsequent API calls
        vision_credentials = assumed_vision_role['Credentials']

        # Use the temporary credentials that AssumeRole returns to make a
        # connection to Amazon S3
        vision_session = boto3.session.Session(
            aws_access_key_id=vision_credentials['AccessKeyId'],
            aws_secret_access_key=vision_credentials['SecretAccessKey'],
            aws_session_token=vision_credentials['SessionToken'], region_name=region
        )

    if not source_role:
        # creates role in source account that has s3 permissions

        source_s3_role = create_source_s3_role(boto3_session=source_session)

        source_sts_client = source_session.client('sts', aws_access_key_id=os.environ['SOURCE_AWS_PUBLIC_KEY'],
                                                  aws_secret_access_key=os.environ['SOURCE_AWS_SECRET_KEY'])
        # Call the assume_role method of the STSConnection object and pass the role
        # ARN and a role session name.

        assumed_source_role = assume_role(sts_client=source_sts_client,
                                          role_arn="arn:aws:iam::{}:role/{}".format(
                                              os.environ['SOURCE_ACCOUNT_NUMBER'], source_s3_role['Role']['RoleName']),
                                          session_name="AssumeRoleSession1"
                                          )

        # From the response that contains the assumed role, get the temporary
        # credentials that can be used to make subsequent API calls
        source_credentials = assumed_source_role['Credentials']

        # Use the temporary credentials that AssumeRole returns to make a
        # connection to Amazon S3
        source_session = boto3.session.Session(
            aws_access_key_id=source_credentials['AccessKeyId'],
            aws_secret_access_key=source_credentials['SecretAccessKey'],
            aws_session_token=source_credentials['SessionToken'], region_name=region
        )

    ###TODO add logic for provided worker and exectutor profiles
    if not worker_profile and executor_role:
        create_emr_roles(vision_session)

    vision_setup(source_session=source_session, vision_session=vision_session, source_bucket='coysu-divina-prototype',
                 worker_profile=worker_profile, executor_role=executor_role, region=region)


create_vision()
