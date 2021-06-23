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
import paramiko
import sys
from botocore.config import Config


####TODO abtract rootish from role jsons - use os.path.expandvars
####TODO abstract bucket names
####TODO setup EC2 SSH keypair when creating cluster


def create_source_s3_role(boto3_session, source_bucket, iam_path='AWS_IAM'):
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), iam_path, 'S3_SOURCE_IAM')) as f:
        policy2_json = os.path.expandvars(json.dumps(json.load(f))).replace('${SOURCE_BUCKET}', source_bucket)
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


def vision_setup(source_bucket, worker_profile, executor_role, vision_session, region, ec2_keyfile,
                 partition_dimensions):
    vision_s3 = vision_session.client('s3')

    try:
        create_bucket(vision_s3, bucket='coysu-divina-prototype-visions',
                      createBucketConfiguration={
                          'LocationConstraint': region
                      })
    except Exception as e:
        raise e
    try:
        source_objects = list_objects(vision_s3, bucket=source_bucket)
    except KeyError as e:
        raise e

    remote_scripts_directory = os.path.join(os.path.abspath('.'), 'remote_scripts')

    for file in os.listdir(remote_scripts_directory):
        path = os.path.join(remote_scripts_directory, file)
        vision_s3.upload_file(
            Filename=path,
            Bucket='coysu-divina-prototype-visions',
            Key='coysu-divina-prototype-{}/remote_scripts/{}'.format(os.environ['VISION_ID'], file)
        )

    file_sizes = []
    for file in source_objects:
        copy_source = {
            'Bucket': source_bucket,
            'Key': file['Key']
        }
        vision_s3.copy_object(CopySource=copy_source, Bucket='coysu-divina-prototype-visions', Key='coysu-divina-prototype-{}/data/{}'.format(os.environ['VISION_ID'], file['Key']))
        size = get_s3_object_size(vision_session, key='coysu-divina-prototype-{}/data/{}'.format(os.environ['VISION_ID'], file['Key']), bucket='coysu-divina-prototype-visions')
        file_sizes.append(size)

    ec2_client = vision_session.client('ec2')

    if ec2_keyfile:
        with open(os.path.join(os.path.expanduser('~'), '.ssh', ec2_keyfile + '.pub')) as f:
            key = ec2_client.import_key_pair(
                KeyName='vision_{}_ec2_key'.format(os.environ['VISION_ID']),
                PublicKeyMaterial=f.read()
            )
            paramiko_key = paramiko.RSAKey.from_private_key_file(
                os.path.join(os.path.expanduser('~'), '.ssh', ec2_keyfile), password=os.environ['KEY_PASS'])
    else:
        key = ec2_client.create_key_pair(
            KeyName='vision_{}_ec2_key'.format(os.environ['VISION_ID'])
        )
        paramiko_key = paramiko.RSAKey.from_private_key(key)

    environment = {'VISION_ID': str(os.environ['VISION_ID']), 'SOURCE_KEYS': [k['Key'] for k in source_objects], 'SOURCE_BUCKET': source_bucket}
    if partition_dimensions:
        environment.update({'PARTITION_DIMENSIONS': partition_dimensions})
    instance = create_ec2_partitioning_instance(vision_session=vision_session, file_size=max(file_sizes), key=key, environment=environment)

    waiter = ec2_client.get_waiter('instance_running')
    waiter.wait(InstanceIds=[i['InstanceId'] for i in instance['Instances']])

    try:
        response = ec2_client.describe_instances(InstanceIds=[i['InstanceId'] for i in instance['Instances']])
        instance = response['Reservations'][0]['Instances'][0]
    except Exception as e:
        raise e

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
            # Here 'ubuntu' is user name and 'instance_ip' is public IP of EC2
        connect_ssh(client, hostname=instance['PublicIpAddress'], username="ec2-user", pkey=paramiko_key)
        commands = ['sudo cp /var/lib/cloud/instance/user-data.txt /home/ec2-user/user-data.json',
                    'sudo yum install unzip -y', 'sudo yum install python3 -y', 'sudo yum install gcc -y',
                    'sudo curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip"',
                    'sudo unzip awscli-bundle.zip',
                    'sudo ./awscli-bundle/install -i /usr/local/aws -b /usr/bin/aws',
                    'sudo aws s3 sync s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/remote_scripts /home/ec2-user/remote_scripts'.format(
                        os.environ['VISION_ID']), 'sudo chown -R ec2-user /home/ec2-user',
                    'python3 -m pip install wheel',
                    'python3 -m pip install -r /home/ec2-user/remote_scripts/requirements.txt',
                    'python3 /home/ec2-user/remote_scripts/partition_data.py']
        for cmd in commands:
            stdin, stdout, stderr = client.exec_command(cmd)
            for line in stdout:
                sys.stdout.write(line)
            for line in stderr:
                sys.stderr.write(line)
            # close the client connection once the job is done
        client.close()

    except Exception as e:
        raise e

        # TODO test that all is well with AWS-generated keys
        # TODO start here get the environment variables into /etc/environment and then restart the shell. just make two client connections.
        # TODO make sure the ec2 instance terminates on completion

        # ec2.stop_instances(InstanceIds=[instance['InstanceId']])

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
                             's3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/remote_scripts'.format(
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
                        "PYSPARK_PYTHON": "/usr/bin/python3",
                        "VISION_ID": os.environ['VISION_ID']
                    },
                }
            ]
        }],
        VisibleToAllUsers=True,
        JobFlowRole=worker_profile,
        ServiceRole=executor_role
    )
    return cluster


def ec2_pricing(pricing_client, region_name, filter_params=None):
    products_params = {'ServiceCode': 'AmazonEC2',
                       'Filters': [
                           {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
                           {'Type': 'TERM_MATCH', 'Field': 'location',
                            'Value': '{}'.format(get_region_name(region_name))},
                           {'Type': 'TERM_MATCH', 'Field': 'capacitystatus', 'Value': 'Used'},
                           {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'},
                           {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'shared'}
                       ]}
    if filter_params:
        products_params['Filters'] = products_params['Filters'] + filter_params
    while True:
        response = pricing_client.get_products(**products_params)
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


def create_ec2_partitioning_instance(vision_session, file_size, key, environment):
    ec2_client = vision_session.client('ec2')
    pricing_client = vision_session.client('pricing', region_name='us-east-1')
    required_gb = math.ceil(file_size * 2 / 1000000000)
    instance_info = [json.loads(p) for p in ec2_pricing(pricing_client, vision_session.region_name) if
                     'memory' in json.loads(p)['product']['attributes'] and 'OnDemand' in json.loads(p)['terms']]
    available_instance_types = [dict(i, **unnest_ec2_price(i)) for i in instance_info if
                                i['product']['attributes']['memory'].split(' ')[0].isdigit()]
    eligible_instance_types = [i for i in available_instance_types if
                               float(i['product']['attributes']['memory'].split(' ')[
                                         0]) >= required_gb and 'Hrs_USD' in i and i['product']['attributes'][
                                                                                       'instanceType'][:2] == 'm5']
    partitioning_instance_type = eligible_instance_types[min(range(len(eligible_instance_types)), key=lambda index:
    eligible_instance_types[index]['Hrs_USD'])]

    instance = ec2_client.run_instances(ImageId='ami-0b223f209b6d4a220', MinCount=1, MaxCount=1,
                                 IamInstanceProfile={'Name': 'EMR_EC2_DefaultRole'},
                                 InstanceType=partitioning_instance_type['product']['attributes']['instanceType'],
                                 KeyName=key['KeyName'],
                                 UserData=json.dumps(environment)
                                 )

    return instance


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


@backoff.on_exception(backoff.expo,
                      paramiko.client.NoValidConnectionsError)
def connect_ssh(client, hostname, username, pkey):
    client.connect(hostname=hostname, username=username, pkey=pkey)
    return client


@backoff.on_exception(backoff.expo,
                      ClientError)
def create_bucket(s3_client, bucket, createBucketConfiguration):
    try:
        s3_client.create_bucket(Bucket=bucket,
                                CreateBucketConfiguration=createBucketConfiguration)
    except ClientError as e:
        if e.response['Error']['Code'] != 'BucketAlreadyOwnedByYou':
            raise e
        else:
            sys.stdout.write('Divina visions s3 bucket already exists...\n')


@backoff.on_exception(backoff.expo,
                      ClientError)
def list_objects(s3_client, bucket):
    objects = s3_client.list_objects(Bucket=bucket)
    if not 'Contents' in objects:
        raise Exception('No objects in source s3 bucket. Upload data to bucket: {} and then retry.'.format(bucket))
    return objects['Contents']


@backoff.on_exception(backoff.expo,
                      ClientError)
def get_bucket_policy(s3_client, bucket):
    return s3_client.get_bucket_policy(Bucket=bucket)


@backoff.on_exception(backoff.expo,
                      ClientError)
def put_bucket_policy(s3_client, bucket, policy):
    return s3_client.put_bucket_policy(Bucket=bucket, Policy=policy)


def create_vision(source_bucket, source_role=None, vision_role=None, worker_profile='EMR_EC2_DefaultRole',
                  executor_role='EMR_DefaultRole', region='us-east-2', ec2_keyfile=None, partition_dimensions=None):
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
            aws_session_token=vision_credentials['SessionToken'], region_name=region,
        )

    if not source_role:
        # creates role in source account that has s3 permissions

        source_s3_role = create_source_s3_role(boto3_session=source_session, source_bucket=source_bucket)

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

        source_s3 = source_session.client('s3')

        try:
            bucket_policy = json.loads(get_bucket_policy(source_s3, bucket=source_bucket)['Policy'])
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucketPolicy':
                bucket_policy = {'Statement': []}
            else:
                raise e

        user_policy = {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::{}:role/{}".format(os.environ['ACCOUNT_NUMBER'],
                                                   vision_role['Role']['RoleName'])
            },
            "Action": [
                "s3:GetBucketLocation",
                "s3:ListBucket",
                "s3:GetObject",
            ],
            "Resource": [
                "arn:aws:s3:::{}".format(source_bucket),
                "arn:aws:s3:::{}/*".format(source_bucket)
            ]
        }

        bucket_policy['Statement'].append(user_policy)

        put_bucket_policy(source_s3, bucket=source_bucket, policy=json.dumps(bucket_policy))

        ###TODO add logic for provided worker and exectutor profiles
    if not worker_profile and executor_role:
        create_emr_roles(vision_session)

    vision_setup(vision_session=vision_session, source_bucket=source_bucket,
                 worker_profile=worker_profile, executor_role=executor_role, region=region, ec2_keyfile=ec2_keyfile,
                 partition_dimensions=partition_dimensions)


create_vision(ec2_keyfile='divina-dev', source_bucket='coysu-divina-prototype-large')
