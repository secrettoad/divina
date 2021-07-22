import boto3
import json
import os
import datetime
from botocore.exceptions import ClientError
import backoff
import subprocess
from pkg_resources import resource_filename, get_distribution
import math
import paramiko
import sys
import io


####TODO abtract rootish from role jsons - use os.path.expandvars

def vision_setup(divina_version, worker_profile, driver_role, vision_session, source_session,
                 ec2_keyfile,
                 vision_role_name,
                 data_definition=None, keep_instances_alive=False, verbosity=0, vision_role=None, source_role=None):
    vision_iam = vision_session.client('iam')
    source_iam = source_session.client('iam')
    vision_sts = vision_session.client('sts')
    source_sts = source_session.client('sts')
    vision_session, source_session = get_sessions(vision_iam=vision_iam, source_iam=source_iam,
                                                  vision_role=vision_role, source_role=source_role,
                                                  vision_sts=vision_sts, source_sts=source_sts
                                                  )
    ###TODO add logic for provided worker and exectutor profiles
    if not worker_profile and driver_role:
        sys.stdout.write('Creating spark driver and executor cloud roles...\n')
        create_emr_roles(vision_session)

    if data_definition:
        sys.stdout.write('Writing data definition...\n')
        with open(os.path.join('..', 'config/data_definition.json'), 'w+') as f:
            json.dump(data_definition, f)
    vision_s3_client = vision_session.client('s3')
    source_s3_client = source_session.client('s3')
    import_data(vision_s3_client=vision_s3_client, source_s3_client=source_s3_client,
                vision_role_name=vision_role_name)
    sys.stdout.write('Uploading artifacts to cloud...\n')
    os.system('export AWS_ACCESS_KEY_ID={}; export AWS_SECRET_ACCESS_KEY={}; aws s3 sync {} s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/divina'.format(os.environ['AWS_PUBLIC_KEY'], os.environ['AWS_SECRET_KEY'], os.path.dirname(os.path.dirname(__file__)), os.environ['VISION_ID']))
    os.system(
        'export AWS_ACCESS_KEY_ID={}; export AWS_SECRET_ACCESS_KEY={}; aws s3 sync {} s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/divina'.format(
            os.environ['AWS_PUBLIC_KEY'], os.environ['AWS_SECRET_KEY'], os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'setup.cfg'),
            os.environ['VISION_ID']))
    os.system(
        'export AWS_ACCESS_KEY_ID={}; export AWS_SECRET_ACCESS_KEY={}; aws s3 sync {} s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/divina'.format(
            os.environ['AWS_PUBLIC_KEY'], os.environ['AWS_SECRET_KEY'], os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'setup.py'),
            os.environ['VISION_ID']))
    sys.stdout.write('Building dataset...\n')
    vision_ec2_client = vision_session.client('ec2')
    vision_pricing_client = vision_session.client('pricing', region_name='us-east-1')
    instance, paramiko_key = create_dataset_ec2(s3_client=vision_s3_client, ec2_client=vision_ec2_client,
                                                pricing_client=vision_pricing_client, ec2_keyfile=ec2_keyfile,
                                                keep_instances_alive=keep_instances_alive, divina_version=divina_version)
    if not build_dataset_ssh(instance=instance, verbosity=verbosity, paramiko_key=paramiko_key):
        if not keep_instances_alive:
            stop_instances(instance_ids=[instance['InstanceId']], ec2_client=vision_ec2_client)
        quit()
    if not keep_instances_alive:
        stop_instances(instance_ids=[instance['InstanceId']], ec2_client=vision_ec2_client)
    sys.stdout.write('Creating forecasts...\n')
    emr_client = vision_session.client('emr')
    emr_cluster = create_modelling_emr(emr_client=emr_client,
                                       worker_profile=worker_profile, driver_role=driver_role,
                                       keep_instances_alive=keep_instances_alive,
                                       ec2_key='vision_{}_ec2_key'.format(os.environ['VISION_ID']))

    run_script_emr(emr_client=emr_client, cluster_id=emr_cluster['JobFlowId'],
                   keep_instances_alive=keep_instances_alive, filename='train.py')
    steps = list_steps(emr_client, emr_cluster['JobFlowId'])
    last_step_id = steps['Steps'][0]['Id']
    emr_waiter = get_emr_waiter(emr_client, 'step_complete')
    emr_waiter.wait(
        ClusterId=emr_cluster['JobFlowId'],
        StepId=last_step_id,
        WaiterConfig={
            "Delay": 30,
            "MaxAttempts": 120
        }
    )


def get_sessions(vision_iam, source_iam, vision_sts, source_sts, vision_role=None, source_role=None):
    if not vision_role:
        sys.stdout.write('Creating Divina cloud role...\n')

        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config',
                               'divina_iam_policy.json')) as f:
            divina_policy = os.path.expandvars(json.dumps(json.load(f)))
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config',
                               'divina_trust_policy.json')) as f:
            vision_role_trust_policy = os.path.expandvars(json.dumps(json.load(f)))

        vision_role = create_role(vision_iam, divina_policy, vision_role_trust_policy, 'divina-vision-role',
                                  'divina-vision-role-policy', 'role for coysu divina')

    assumed_vision_role = assume_role(sts_client=vision_sts,
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
            aws_session_token=vision_credentials['SessionToken'], region_name=vision_sts._client_config.region_name,
    )

    if not source_role:
        # creates role in source account that has s3 permissions
        sys.stdout.write('Connecting import cloud role...\n')

        with open(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config',
                             'import_iam_policy.json')) as f:
            source_policy = os.path.expandvars(json.dumps(json.load(f))).replace('${IMPORT_BUCKET}', os.environ['IMPORT_BUCKET'])
        with open(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config',
                             'import_trust_policy.json')) as f:
            source_role_trust_policy = os.path.expandvars(json.dumps(json.load(f)))

        source_s3_role = create_role(source_iam, source_policy, source_role_trust_policy, 'divina-source-role',
                                     'divina-source-role-policy',
                                     'policy for coysu divina to assume when importing datasets')

    assumed_source_role = assume_role(sts_client=source_sts,
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
        aws_session_token=source_credentials['SessionToken'], region_name=source_sts._client_config.region_name
    )

    return vision_session, source_session


def create_dataset_ec2(divina_version, s3_client, ec2_client, pricing_client, ec2_keyfile=None,
                       keep_instances_alive=False):
    file_sizes = []
    keys = []
    for o in list_objects(s3_client=s3_client, bucket=os.environ['DIVINA_BUCKET'],
                          prefix='coysu-divina-prototype-{}/data/'.format(os.environ['VISION_ID'])):
        file_sizes.append(get_s3_object_size(s3_client=s3_client, key=o['Key'], bucket=os.environ['DIVINA_BUCKET']))
        keys.append(o['Key'])

    if ec2_keyfile:
        with open(os.path.join(os.path.expanduser('~'), '.ssh', ec2_keyfile + '.pub')) as f:
            key = import_key_pair(
                key_name='vision_{}_ec2_key'.format(os.environ['VISION_ID']),
                public_key_material=f.read(), ec2_client=ec2_client
            )
            paramiko_key = paramiko.RSAKey.from_private_key_file(
                os.path.join(os.path.expanduser('~'), '.ssh', ec2_keyfile), password=os.environ['KEY_PASS'])
    else:
        key = ec2_client.create_key_pair(
            KeyName='vision_{}_ec2_key'.format(os.environ['VISION_ID'])
        )
        paramiko_key = paramiko.RSAKey.from_private_key(io.StringIO(key['KeyMaterial']))

    environment = {
        'ENVIRONMENT': {'VISION_ID': str(os.environ['VISION_ID']),
                        'DIVINA_BUCKET': os.environ['DIVINA_BUCKET'], 'DIVINA_VERSION': divina_version}}

    security_groups = describe_security_groups(
        filters=[
            dict(Name='group-name', Values=['divina-ssh'])
        ], ec2_client=ec2_client
    )
    ip_permissions = [
        {'IpRanges': [
            {
                'CidrIp': '0.0.0.0/0',
                'Description': 'divina-ip'
            },
        ],

            'IpProtocol': 'tcp',
            'FromPort': 22,
            'ToPort': 22,
        },

    ]
    if len(security_groups['SecurityGroups']) > 0:
        security_group = security_groups['SecurityGroups'][0]
        if not ip_permissions[0]['IpRanges'][0]['CidrIp'] in [ipr['CidrIp'] for s in security_group['IpPermissions'] for
                                                              ipr in s['IpRanges']
                                                              if all(
                [s[k] == ip_permissions[0][k] for k in ['FromPort', 'ToPort', 'IpProtocol']])]:
            authorize_security_group_ingress(
                group_id=security_group['GroupId'],
                ip_permissions=ip_permissions, ec2_client=ec2_client

            )

    else:
        vpc_id = ec2_client.describe_vpcs()['Vpcs'][0]['VpcId']

        security_group = create_security_group(
            description='Security group for allowing SSH access to partitioning VM for Coysu Divina',
            group_name='divina-ssh',
            vpc_id=vpc_id, ec2_client=ec2_client
        )
        authorize_security_group_ingress(
            group_id=security_group['GroupId'],
            ip_permissions=ip_permissions, ec2_client=ec2_client

        )

    required_ram = math.ceil(max(file_sizes) * 10 / 1000000000)
    required_disk = math.ceil(max(file_sizes) / 1000000000) + 3
    instance_info = [json.loads(p) for p in ec2_pricing(pricing_client, ec2_client._client_config.region_name) if
                     'memory' in json.loads(p)['product']['attributes'] and 'OnDemand' in json.loads(p)['terms']]
    available_instance_types = [dict(i, **unnest_ec2_price(i)) for i in instance_info if
                                i['product']['attributes']['memory'].split(' ')[0].isdigit()]
    eligible_instance_types = [i for i in available_instance_types if
                               float(i['product']['attributes']['memory'].split(' ')[
                                         0]) >= required_ram and 'Hrs_USD' in i and i['product']['attributes'][
                                                                                        'instanceType'][:2] == 'm5']
    partitioning_instance_type = eligible_instance_types[min(range(len(eligible_instance_types)), key=lambda index:
    eligible_instance_types[index]['Hrs_USD'])]

    instance = ec2_client.run_instances(ImageId='ami-0b223f209b6d4a220', MinCount=1, MaxCount=1,
                                        IamInstanceProfile={'Name': 'EMR_EC2_DefaultRole'},
                                        InstanceType=partitioning_instance_type['product']['attributes'][
                                            'instanceType'],
                                        KeyName=key['KeyName'],
                                        UserData=json.dumps(environment),
                                        BlockDeviceMappings=[
                                            {"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": required_disk}}],
                                        SecurityGroupIds=[
                                            security_group['GroupId']
                                        ]
                                        )

    try:
        waiter = get_ec2_waiter('instance_running', ec2_client=ec2_client)
        waiter.wait(InstanceIds=[i['InstanceId'] for i in instance['Instances']])
        response = describe_instances(instance_ids=[i['InstanceId'] for i in instance['Instances']],
                                      ec2_client=ec2_client)
        instance = response['Reservations'][0]['Instances'][0]

    except Exception as e:
        if not keep_instances_alive:
            stop_instances(instance_ids=[instance['InstanceId']], ec2_client=ec2_client)
        raise e
    return instance, paramiko_key


def build_dataset_ssh(instance, verbosity, paramiko_key):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_ssh(client, hostname=instance['PublicIpAddress'], username="ec2-user", pkey=paramiko_key)
    commands = ['sudo cp /var/lib/cloud/instance/user-data.txt /home/ec2-user/user-data.json',
                'sudo yum install unzip -y', 'sudo yum install python3 -y', 'sudo yum install gcc -y',
                'sudo curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"',
                'sudo unzip awscliv2.zip',
                'sudo sudo ./aws/install -i /usr/local/aws-cli -b /usr/local/bin',
                'aws codeartifact login --tool pip --domain coysu --repository divina',
                'pip install divina=={} --extra-index-url https://www.pypi.org/simple'.format(pkg_resources.get_distribution('divina').version),
                'aws s3 cp s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/data_definition.json /home/ec2-user/data_definition.json'.format(
                    os.environ['VISION_ID']),
                'sudo chown -R ec2-user /home/ec2-user',
                'divina forecast build_dataset /home/ec2-user/data']
    for cmd in commands:
        stdin, stdout, stderr = client.exec_command(cmd)
        if verbosity > 0:
            for line in stdout:
                sys.stdout.write(line)
        if verbosity > 2:
            for line in stderr:
                sys.stderr.write(line)
        exit_status = stdout.channel.recv_exit_status()
        if not exit_status == 0:
            client.close()
            return False
    client.close()
    return True


def upload_scripts(s3_client, bucket):
    upload_dirs = {'divina': os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}

    for d in upload_dirs:
        if not d == '/':
            key = '{}/{}'.format('coysu-divina-prototype-{}'.format(os.environ['VISION_ID']), d)
        else:
            key = '{}'.format('coysu-divina-prototype-{}'.format(os.environ['VISION_ID']))
        for file in os.listdir(upload_dirs[d]):
            path = os.path.join(upload_dirs[d], file)
            with open(path) as f:
                upload_file(s3_client=s3_client,
                            bucket=bucket,
                            key=os.path.join(key, file),
                            body=f.read())


def import_data(vision_s3_client, source_s3_client, vision_role_name):
    user_policy = {
        "Effect": "Allow",
        "Principal": {
            "AWS": "arn:aws:iam::{}:role/{}".format(os.environ['ACCOUNT_NUMBER'],
                                                    vision_role_name)
        },
        "Action": [
            "s3:GetBucketLocation",
            "s3:ListBucket",
            "s3:GetObject"
        ],
        "Resource": [
            "arn:aws:s3:::{}".format(os.environ['IMPORT_BUCKET']),
            "arn:aws:s3:::{}/*".format(os.environ['IMPORT_BUCKET'])
        ]
    }

    sys.stdout.write('Granting Divina access to imported data...\n')
    try:
        bucket_policy = json.loads(get_bucket_policy(source_s3_client, bucket=os.environ['IMPORT_BUCKET'])['Policy'])
        if not user_policy in bucket_policy['Statement']:
            bucket_policy['Statement'].append(user_policy)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucketPolicy':
            bucket_policy = {'Statement': [user_policy]}
        else:
            raise e

    put_bucket_policy(source_s3_client, bucket=os.environ['IMPORT_BUCKET'], policy=json.dumps(bucket_policy))

    sys.stdout.write('Creating Divina cloud storage...\n')
    try:
        create_bucket(vision_s3_client, bucket='coysu-divina-prototype-visions',
                      createBucketConfiguration={
                          'LocationConstraint': vision_s3_client._client_config.region_name
                      })

    except Exception as e:
        raise e
    try:
        source_objects = list_objects(source_s3_client, bucket=os.environ['IMPORT_BUCKET'])
    except KeyError as e:
        raise e

    sys.stdout.write('Importing data...\n')
    for file in source_objects:
        copy_object(copy_source={
            'Bucket': os.environ['IMPORT_BUCKET'],
            'Key': file['Key']
        }, bucket='coysu-divina-prototype-visions',
            key='coysu-divina-prototype-{}/data/{}'.format(os.environ['VISION_ID'], file['Key']),
            s3_client=vision_s3_client)

    return None


def create_emr_roles(boto3_session):
    iam_client = boto3_session.client('iam')

    if not all(x in [r['RoleName'] for r in iam_client.list_roles()['Roles']] for x in
               ['EMR_EC2_DefaultRole', 'EMR_DefaultRole']):
        subprocess.run(['aws', 'emr', 'create-default-roles'])

    return iam_client.list_roles()


def create_modelling_emr(emr_client, worker_profile='EMR_EC2_DefaultRole',
                         driver_role='EMR_DefaultRole', keep_instances_alive=False, ec2_key=None):
    if keep_instances_alive:
        on_failure = 'CANCEL_AND_WAIT'
    else:
        on_failure = 'TERMINATE_CLUSTER'
    with open(os.path.join('..', 'divina/config/emr_config.json'), 'r') as f:
        if ec2_key:
            emr_config = json.loads(
                os.path.expandvars(json.dumps(json.load(f)).replace('${WORKER_PROFILE}', worker_profile).replace(
                    '${DRIVER_ROLE}',
                    driver_role).replace(
                    '${EMR_ON_FAILURE}',
                    on_failure)).replace(
                    '${EC2_KEYNAME}', ec2_key))
        else:
            emr_config = json.loads(
                os.path.expandvars(json.dumps(json.load(f)).replace('${WORKER_PROFILE}', worker_profile).replace(
                    '${DRIVER_ROLE}',
                    driver_role).replace(
                    '${EMR_ON_FAILURE}',
                    on_failure)))

    cluster = emr_client.run_job_flow(**emr_config['emr_config'])
    steps = list_steps(emr_client, cluster['JobFlowId'])
    last_step_id = steps['Steps'][0]['Id']
    emr_waiter = get_emr_waiter(emr_client, 'step_complete')
    emr_waiter.wait(
        ClusterId=cluster['JobFlowId'],
        StepId=last_step_id,
        WaiterConfig={
            "Delay": 30,
            "MaxAttempts": 120
        }
    )
    return cluster


def run_script_emr(emr_client, cluster_id, keep_instances_alive, filename):
    if keep_instances_alive:
        on_failure = 'CANCEL_AND_WAIT'
    else:
        on_failure = 'TERMINATE_CLUSTER'
    steps = emr_client.add_job_flow_steps(
        JobFlowId=cluster_id,
        Steps=[
            {
                "Name": "vision_script",
                "ActionOnFailure": on_failure,
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "/home/hadoop/spark_scripts/{}".format(filename)
                    ]
                }
            }
        ]
    )
    emr_waiter = get_emr_waiter(emr_client, 'step_complete')
    emr_waiter.wait(
        ClusterId=cluster_id,
        StepId=steps['StepIds'][0],
        WaiterConfig={
            "Delay": 60,
            "MaxAttempts": 120
        }
    )
    return steps


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
        response = get_products(pricing_client, products_params)
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


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def create_security_group(description, group_name, vpc_id, ec2_client):
    return ec2_client.create_security_group(
        Description=description,
        GroupName=group_name,
        VpcId=vpc_id
    )


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def authorize_security_group_ingress(group_id, ip_permissions, ec2_client):
    ec2_client.authorize_security_group_ingress(
        GroupId=group_id,
        IpPermissions=ip_permissions

    )


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def describe_security_groups(filters, ec2_client):
    return ec2_client.describe_security_groups(
        Filters=filters
    )


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def list_steps(emr_client, cluster_id):
    return emr_client.list_steps(ClusterId=cluster_id)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def get_ec2_waiter(step, ec2_client):
    return ec2_client.get_waiter(step)


@backoff.on_exception(backoff.expo,
                      ClientError, max_tries=10)
def copy_object(copy_source, bucket, key, s3_client):
    return s3_client.copy_object(CopySource=copy_source, Bucket=bucket,
                                 Key=key)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def describe_instances(instance_ids, ec2_client):
    return ec2_client.describe_instances(InstanceIds=instance_ids)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def import_key_pair(
        key_name,
        public_key_material, ec2_client
):
    return ec2_client.import_key_pair(KeyName=key_name, PublicKeyMaterial=public_key_material)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def stop_instances(instance_ids, ec2_client):
    ec2_client.stop_instances(InstanceIds=instance_ids)


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def get_emr_waiter(emr_client, step):
    waiter = emr_client.get_waiter(step)
    return waiter


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def delete_emr_cluster(emr_cluster, emr_client):
    emr_client.terminate_job_flows(JobFlowIds=[emr_cluster['JobFlowId']])


@backoff.on_exception(backoff.expo,
                      ClientError, max_time=30)
def get_products(pricing_client, products_params):
    return pricing_client.get_products(**products_params)


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
                      ClientError, max_tries=10)
def list_objects(s3_client, bucket, prefix=None):
    if prefix:
        objects = s3_client.list_objects(Bucket=bucket, Prefix=prefix)
    else:
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
def create_role(iam_client, policy_document, trust_policy_document, role_name, policy_name, description):
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


def create_vision(import_bucket, divina_version, source_role=None, vision_role=None, worker_profile='EMR_EC2_DefaultRole',
                  driver_role='EMR_DefaultRole', region='us-east-2', ec2_keyfile=None, data_definition=None,
                  keep_instances_alive=False, verbosity=0):
    os.environ['VISION_ID'] = str(round(datetime.datetime.now().timestamp()))
    os.environ['DIVINA_BUCKET'] = 'coysu-divina-prototype-visions'
    os.environ['IMPORT_BUCKET'] = import_bucket

    sys.stdout.write('Authenticating to the cloud...\n')
    source_session = boto3.session.Session(aws_access_key_id=os.environ['SOURCE_AWS_PUBLIC_KEY'],
                                           aws_secret_access_key=os.environ['SOURCE_AWS_SECRET_KEY'],
                                           region_name=region)

    vision_session = boto3.session.Session(aws_access_key_id=os.environ['AWS_PUBLIC_KEY'],
                                           aws_secret_access_key=os.environ['AWS_SECRET_KEY'], region_name=region)

    ###current scope is that all supplied files are either a single endogenous schema OR an exogenous signal that can be joined to the endogenous schema
    vision_setup(vision_session=vision_session, source_session=source_session,
                 vision_role=vision_role, source_role=source_role, worker_profile=worker_profile,
                 driver_role=driver_role, vision_role_name='divina-vision-role',
                 ec2_keyfile=ec2_keyfile,
                 data_definition=data_definition, keep_instances_alive=keep_instances_alive, verbosity=verbosity, divina_version=divina_version)


create_vision(import_bucket='coysu-divina-prototype-small', ec2_keyfile='divina-dev', verbosity=3)