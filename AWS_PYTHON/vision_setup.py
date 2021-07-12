import boto3
import json
import os
import datetime
from botocore.exceptions import ClientError
import backoff
import subprocess
from pkg_resources import resource_filename
import math
import paramiko
import sys
import time


####TODO abtract rootish from role jsons - use os.path.expandvars
####TODO abstract bucket names


def vision_setup(source_bucket, worker_profile, driver_role, vision_session, region, ec2_keyfile,
                 data_definition=None, keep_instances_alive=False):
    vision_s3 = vision_session.client('s3')

    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')

    if data_definition:
        sys.stdout.write('Writing data definition...\n')
        with open(os.path.join('.', 'tmp/data_definition.json'), 'w+') as f:
            json.dump(data_definition, f)

    sys.stdout.write('Creating Divina cloud storage...\n')
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

    sys.stdout.write('Uploading artifacts to cloud...\n')
    upload_directories = {}
    upload_directories.update({'remote_scripts': os.path.join(os.path.abspath('.'), 'remote_scripts')})
    upload_directories.update({'spark_scripts': os.path.join(os.path.abspath('.'), 'spark_scripts')})
    upload_directories.update({'/': os.path.join(os.path.abspath('.'), 'tmp')})

    for d in upload_directories:
        if not d == '/':
            key = 'coysu-divina-prototype-{}/{}'.format(os.environ['VISION_ID'], d)
        else:
            key = 'coysu-divina-prototype-{}'.format(os.environ['VISION_ID'])
        for file in os.listdir(upload_directories[d]):
            path = os.path.join(upload_directories[d], file)
            with open(path) as f:
                upload_file(s3_client=vision_s3,
                            bucket='coysu-divina-prototype-visions',
                            key=os.path.join(key, file),
                            body=f.read())

    file_sizes = []
    sys.stdout.write('Importing data...\n')
    for file in source_objects:
        copy_source = {
            'Bucket': source_bucket,
            'Key': file['Key']
        }
        vision_s3.copy_object(CopySource=copy_source, Bucket='coysu-divina-prototype-visions',
                              Key='coysu-divina-prototype-{}/data/{}'.format(os.environ['VISION_ID'], file['Key']))
        size = get_s3_object_size(vision_s3,
                                  key='coysu-divina-prototype-{}/data/{}'.format(os.environ['VISION_ID'], file['Key']),
                                  bucket='coysu-divina-prototype-visions')
        file_sizes.append(size)

    sys.stdout.write('Building dataset...\n')
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

    environment = {
        'ENVIRONMENT': {'VISION_ID': str(os.environ['VISION_ID']), 'SOURCE_KEYS': [k['Key'] for k in source_objects],
                        'SOURCE_BUCKET': source_bucket}}
    instance = create_ec2_partitioning_instance(vision_session=vision_session, file_size=max(file_sizes), key=key,
                                                environment=environment)

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
                        os.environ['VISION_ID']),
                    'sudo aws s3 cp s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/data_definition.json /home/ec2-user/data_definition.json'.format(
                        os.environ['VISION_ID']),
                    'sudo chown -R ec2-user /home/ec2-user',
                    'python3 -m pip install wheel',
                    'python3 -m pip install -r /home/ec2-user/remote_scripts/requirements.txt',
                    'python3 /home/ec2-user/remote_scripts/partition_data.py']
        for cmd in commands:
            stdin, stdout, stderr = client.exec_command(cmd)
            for line in stdout:
                sys.stdout.write(line)
            for line in stderr:
                sys.stderr.write(line)
                ###TODO GET THIS TO WORK BELOW
            exit_status = stdout.channel.recv_exit_status()
            if not exit_status == 0:
                client.close()
                if not keep_instances_alive:
                    ec2_client.stop_instances(InstanceIds=[instance['InstanceId']])
                quit()
            # close the client connection once the job is done
        client.close()

    except Exception as e:
        raise e

    if not keep_instances_alive:
        ec2_client.stop_instances(InstanceIds=[instance['InstanceId']])

    sys.stdout.write('Creating forecasts...\n')
    emr_client = vision_session.client('emr')
    emr_cluster = run_vision_and_validation(emr_client=emr_client,
                                worker_profile=worker_profile, driver_role=driver_role,
                                keep_instances_alive=keep_instances_alive,
                                ec2_key='vision_{}_ec2_key'.format(os.environ['VISION_ID']))

    emr_waiter = emr_client.get_waiter('step_complete')
    emr_waiter.wait(
        ClusterId='the-cluster-id',
        StepId='the-step-id',
        WaiterConfig={
            "Delay": 30,
            "MaxAttempts": 10
        }
    )

    delete_emr_cluster(emr_cluster, vision_session)


def create_emr_roles(boto3_session):
    iam_client = boto3_session.client('iam')

    if not all(x in [r['RoleName'] for r in iam_client.list_roles()['Roles']] for x in
               ['EMR_EC2_DefaultRole', 'EMR_DefaultRole']):
        subprocess.run(['aws', 'emr', 'create-default-roles'])

    return iam_client.list_roles()


def delete_emr_cluster(emr_cluster, boto3_session):
    emr_client = boto3_session.client('emr')
    emr_client.terminate_job_flows(JobFlowIds=[emr_cluster['JobFlowId']])


def run_vision_and_validation(emr_client, worker_profile='EMR_EC2_DefaultRole',
                      driver_role='EMR_DefaultRole', keep_instances_alive=False, ec2_key=None):
    if keep_instances_alive:
        on_failure = 'CANCEL_AND_WAIT'
    else:
        on_failure = 'TERMINATE_CLUSTER'
    environment = {"PYSPARK_PYTHON": "/usr/bin/python3", "VISION_ID": os.environ['VISION_ID']}

    cluster = emr_client.run_job_flow(
        Name="divina-cluster-{}".format(os.environ['VISION_ID']),
        ReleaseLabel='emr-6.2.0',
        Instances={
            'KeepJobFlowAliveWhenNoSteps': keep_instances_alive,
            'TerminationProtected': False,
            'InstanceGroups': [
                {
                    'InstanceRole': 'TASK',
                    'InstanceType': 'c4.xlarge',
                    'InstanceCount': 3
                },
                {
                    'InstanceRole': 'MASTER',
                    'InstanceType': 'c4.xlarge',
                    'InstanceCount': 1
                },
                {
                    'InstanceRole': 'CORE',
                    'InstanceType': 'c4.xlarge',
                    'InstanceCount': 1
                }
            ],
            'Ec2KeyName': ec2_key,

        },
        LogUri='s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/logs/emr_logs'.format(
            os.environ['VISION_ID']),
        Applications=[
            {
                'Name': 'Spark'
            }
        ],
        Steps=[
            {
                'Name': 'setup_state_pusher',
                'ActionOnFailure': on_failure,
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': ['state-pusher-script']
                }
            },
            {
                'Name': 'setup_s3_sync_scripts',
                'ActionOnFailure': on_failure,
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': ['aws', 's3', 'sync',
                             's3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/spark_scripts'.format(
                                 os.environ['VISION_ID']),
                             '/home/hadoop/spark_scripts']
                }
            },
            {
                'Name': 'setup_s3_sync_data',
                'ActionOnFailure': on_failure,
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': ['aws', 's3', 'cp',
                             's3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/data_definition.json'.format(
                                 os.environ['VISION_ID']),
                             '/home/hadoop/data_definition.json']
                }
            },
            {
                'Name': 'vision_script',
                'ActionOnFailure': on_failure,
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': ['spark-submit', '/home/hadoop/spark_scripts/predict.py']
                }
            },
            {
                'Name': 'validation_script',
                'ActionOnFailure': on_failure,
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': ['spark-submit', '/home/hadoop/spark_scripts/validate.py']
                }
            }
        ],
        Configurations=[{
            "Classification": "spark-env",
            "Configurations": [
                {
                    "Classification": "export",
                    "Properties": environment
                }
            ]
        }],
        VisibleToAllUsers=True,
        JobFlowRole=worker_profile,
        ServiceRole=driver_role
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
    required_ram = math.ceil(file_size * 10 / 1000000000)
    required_disk = math.ceil(file_size / 1000000000) + 3
    instance_info = [json.loads(p) for p in ec2_pricing(pricing_client, vision_session.region_name) if
                     'memory' in json.loads(p)['product']['attributes'] and 'OnDemand' in json.loads(p)['terms']]
    available_instance_types = [dict(i, **unnest_ec2_price(i)) for i in instance_info if
                                i['product']['attributes']['memory'].split(' ')[0].isdigit()]
    eligible_instance_types = [i for i in available_instance_types if
                               float(i['product']['attributes']['memory'].split(' ')[
                                         0]) >= required_ram and 'Hrs_USD' in i and i['product']['attributes'][
                                                                                       'instanceType'][:2] == 'm5']
    partitioning_instance_type = eligible_instance_types[min(range(len(eligible_instance_types)), key=lambda index:
    eligible_instance_types[index]['Hrs_USD'])]

    ###TODO dynamically size EBS with  and then mount to filesystem and write data

    instance = ec2_client.run_instances(ImageId='ami-0b223f209b6d4a220', MinCount=1, MaxCount=1,
                                        IamInstanceProfile={'Name': 'EMR_EC2_DefaultRole'},
                                        InstanceType=partitioning_instance_type['product']['attributes'][
                                            'instanceType'],
                                        KeyName=key['KeyName'],
                                        UserData=json.dumps(environment),
                                        BlockDeviceMappings=[{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": required_disk}}]
                                        )

    return instance


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def get_s3_object_size(s3_client, key, bucket):
    response = s3_client.head_object(Bucket=bucket, Key=key)
    size = response['ContentLength']
    return size


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
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
                      ClientError, max_time=30)
def create_bucket(s3_client, bucket, createBucketConfiguration):
    try:
        s3_client.create_bucket(Bucket=bucket,
                                CreateBucketConfiguration=createBucketConfiguration)
    except ClientError as e:
        if e.response['Error']['Code'] != 'BucketAlreadyOwnedByYou':
            raise e
        else:
            sys.stdout.write('Cloud storage already exists...\n')


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def list_objects(s3_client, bucket):
    objects = s3_client.list_objects(Bucket=bucket)
    if not 'Contents' in objects:
        raise Exception('No data to import. Upload data to bucket: {} and then retry.'.format(bucket))
    return objects['Contents']


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def upload_file(s3_client, bucket, key, body):
    return s3_client.put_object(
        Bucket=bucket,
        Key=key, Body=body)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def get_bucket_policy(s3_client, bucket):
    return s3_client.get_bucket_policy(Bucket=bucket)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def put_bucket_policy(s3_client, bucket, policy):
    return s3_client.put_bucket_policy(Bucket=bucket, Policy=policy)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def create_role(iam_client, policy_document, trust_policy_document, role_name, policy_name, description,
                iam_path='AWS_IAM'):
    try:
        role = iam_client.get_role(RoleName=role_name)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            sys.stdout.write('Divina role: {} not found. Creating now.\n'.format(role_name))
            role = iam_client.create_role(
                Path='/',
                RoleName=role_name,
                AssumeRolePolicyDocument=trust_policy_document,
                Description=description
            )
        else:
            raise e

    try:
        iam_client.get_policy(
            PolicyArn='arn:aws:iam::{}:policy/{}'.format(os.environ['ACCOUNT_NUMBER'], policy_name)
        )
        policy_version = iam_client.create_policy_version(
            PolicyArn='arn:aws:iam::{}:policy/{}'.format(os.environ['ACCOUNT_NUMBER'], policy_name),
            PolicyDocument=policy_document,
            SetAsDefault=True
        )

        iam_client.attach_role_policy(
            RoleName=role_name, PolicyArn='arn:aws:iam::{}:policy/{}'.format(os.environ['ACCOUNT_NUMBER'], policy_name))

        iam_client.delete_policy_version(
            PolicyArn='arn:aws:iam::{}:policy/{}'.format(os.environ['ACCOUNT_NUMBER'], policy_name),
            VersionId='v{}'.format(int(policy_version['PolicyVersion']['VersionId'][1:]) - 1)
        )

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            sys.stdout.write('Divina policy: {} not found. Creating now.\n'.format(policy_name))
            iam_client.create_policy(
                PolicyName=policy_name,
                PolicyDocument=policy_document
            )
        else:
            raise e

    iam_client.attach_role_policy(
        RoleName=role_name, PolicyArn='arn:aws:iam::{}:policy/{}'.format(os.environ['ACCOUNT_NUMBER'], policy_name))

    return role


def create_vision(source_bucket, source_role=None, vision_role=None, worker_profile='EMR_EC2_DefaultRole',
                  driver_role='EMR_DefaultRole', region='us-east-2', ec2_keyfile=None, data_definition=None,
                  keep_instances_alive=False):
    os.environ['VISION_ID'] = str(round(datetime.datetime.now().timestamp()))

    sys.stdout.write('Authenticating to the cloud...\n')
    source_session = boto3.session.Session(aws_access_key_id=os.environ['SOURCE_AWS_PUBLIC_KEY'],
                                           aws_secret_access_key=os.environ['SOURCE_AWS_SECRET_KEY'],
                                           region_name=region)

    vision_session = boto3.session.Session(aws_access_key_id=os.environ['AWS_PUBLIC_KEY'],
                                           aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name=region)

    if not vision_role:
        sys.stdout.write('Creating Divina cloud role...\n')
        vision_iam = vision_session.client('iam')

        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), os.environ['IAM_PATH'], 'DIVINA_IAM')) as f:
            divina_policy = os.path.expandvars(json.dumps(json.load(f)))
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), os.environ['IAM_PATH'],
                               'DIVINA_ROLE_TRUST_POLICY')) as f:
            vision_role_trust_policy = os.path.expandvars(json.dumps(json.load(f)))

        vision_role = create_role(vision_iam, divina_policy, vision_role_trust_policy, 'divina-vision-role',
                                  'divina-vision-role-policy', 'role for coysu divina')

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
        sys.stdout.write('Connecting import cloud role...\n')
        source_iam = source_session.client('iam')

        with open(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), os.environ['IAM_PATH'], 'S3_SOURCE_IAM')) as f:
            source_policy = os.path.expandvars(json.dumps(json.load(f))).replace('${SOURCE_BUCKET}', source_bucket)
        with open(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), os.environ['IAM_PATH'],
                             'S3_SOURCE_ROLE_TRUST_POLICY')) as f:
            source_role_trust_policy = os.path.expandvars(json.dumps(json.load(f)))

        source_s3_role = create_role(source_iam, source_policy, source_role_trust_policy, 'divina-source-role',
                                     'divina-source-role-policy',
                                     'policy for coysu divina to assume when importing datasets')

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

        sys.stdout.write('Granting Divina access to imported data...\n')
        try:
            bucket_policy = json.loads(get_bucket_policy(source_s3, bucket=source_bucket)['Policy'])
            if not user_policy in bucket_policy['Statement']:
                bucket_policy['Statement'].append(user_policy)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucketPolicy':
                bucket_policy = {'Statement': [user_policy]}
            else:
                raise e

        put_bucket_policy(source_s3, bucket=source_bucket, policy=json.dumps(bucket_policy))

        ###TODO add logic for provided worker and exectutor profiles
    if not worker_profile and driver_role:
        sys.stdout.write('Creating spark driver and executor cloud roles...\n')
        create_emr_roles(vision_session)

    ###current scope is that all supplied files are either a single endogenous schema OR an exogenous signal that can be joined to the endogenous schema
    vision_setup(vision_session=vision_session, source_bucket=source_bucket,
                 worker_profile=worker_profile, driver_role=driver_role, region=region, ec2_keyfile=ec2_keyfile,
                 keep_instances_alive=keep_instances_alive)

num_retries = 2
n = 0
while n < num_retries:
    try:
        n += 1
        create_vision(ec2_keyfile='divina-dev', source_bucket='coysu-divina-prototype-large',
              keep_instances_alive=True)
    except:
        time.sleep(10**n)

