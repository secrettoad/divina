import boto3
import os
import datetime
import subprocess
from pkg_resources import resource_filename, get_distribution
import math
import paramiko
import io
import sys
import json
from ..aws import aws_backoff
from .import_data import import_data
import backoff


####TODO abtract rootish from role jsons - use os.path.expandvars

def vision_setup(divina_version, worker_profile, driver_role, vision_session, source_session,
                 ec2_keyfile,
                 vision_role_name,
                 data_definition=None, keep_instances_alive=False, verbosity=0, vision_role=None, source_role=None, divina_pip_arguments=None):
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

    aws_backoff.upload_file(s3_client=vision_s3_client,
                bucket=os.environ['DIVINA_BUCKET'],
                key='coysu-divina-prototype-{}/data_definition.json'.format(os.environ['VISION_ID']),
                body=io.StringIO(json.dumps(data_definition)).read())

    os.system('aws s3 sync {} s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/data_definition.json'.format(os.path.join('..', 'config/data_definition.json'), os.environ['VISION_ID']))

    import_data(vision_s3_client=vision_s3_client, source_s3_client=source_s3_client,
                vision_role_name=vision_role_name)
    sys.stdout.write('Building dataset...\n')
    vision_ec2_client = vision_session.client('ec2')
    vision_pricing_client = vision_session.client('pricing', region_name='us-east-1')
    instance, paramiko_key = create_dataset_ec2(s3_client=vision_s3_client, ec2_client=vision_ec2_client,
                                                pricing_client=vision_pricing_client, ec2_keyfile=ec2_keyfile,
                                                keep_instances_alive=keep_instances_alive,
                                                divina_version=divina_version)
    if not build_dataset_ssh(instance=instance, verbosity=verbosity, paramiko_key=paramiko_key, divina_pip_arguments=divina_pip_arguments):
        if not keep_instances_alive:
            aws_backoff.stop_instances(instance_ids=[instance['InstanceId']], ec2_client=vision_ec2_client)
        quit()
    if not keep_instances_alive:
        aws_backoff.stop_instances(instance_ids=[instance['InstanceId']], ec2_client=vision_ec2_client)
    sys.stdout.write('Creating forecasts...\n')
    emr_client = vision_session.client('emr')
    emr_cluster = create_modelling_emr(emr_client=emr_client,
                                       worker_profile=worker_profile, driver_role=driver_role,
                                       keep_instances_alive=keep_instances_alive,
                                       ec2_key='vision_{}_ec2_key'.format(os.environ['VISION_ID']))

    run_command_emr(emr_client=emr_client, cluster_id=emr_cluster['JobFlowId'],
                   keep_instances_alive=keep_instances_alive, args=['divina', 'train', '--data_definition', '/home/hadoop/data_definition.json', ''])
    steps = aws_backoff.list_steps(emr_client, emr_cluster['JobFlowId'])
    last_step_id = steps['Steps'][0]['Id']
    emr_waiter = aws_backoff.get_emr_waiter(emr_client, 'step_complete')
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

        vision_role = aws_backoff.create_role(vision_iam, divina_policy, vision_role_trust_policy, 'divina-vision-role',
                                  'divina-vision-role-policy', 'role for coysu divina')

    assumed_vision_role = aws_backoff.assume_role(sts_client=vision_sts,
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
            source_policy = os.path.expandvars(json.dumps(json.load(f))).replace('${IMPORT_BUCKET}',
                                                                                 os.environ['IMPORT_BUCKET'])
        with open(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config',
                             'import_trust_policy.json')) as f:
            source_role_trust_policy = os.path.expandvars(json.dumps(json.load(f)))

        source_s3_role = aws_backoff.create_role(source_iam, source_policy, source_role_trust_policy, 'divina-source-role',
                                     'divina-source-role-policy',
                                     'policy for coysu divina to assume when importing datasets')

    assumed_source_role = aws_backoff.assume_role(sts_client=source_sts,
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
    for o in aws_backoff.list_objects(s3_client=s3_client, bucket=os.environ['DIVINA_BUCKET'],
                          prefix='coysu-divina-prototype-{}/data/'.format(os.environ['VISION_ID'])):
        file_sizes.append(aws_backoff.get_s3_object_size(s3_client=s3_client, key=o['Key'], bucket=os.environ['DIVINA_BUCKET']))
        keys.append(o['Key'])

    if ec2_keyfile:
        with open(os.path.join(os.path.expanduser('~'), '.ssh', ec2_keyfile + '.pub')) as f:
            key = aws_backoff.import_key_pair(
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

    security_groups = aws_backoff.describe_security_groups(
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
            aws_backoff.authorize_security_group_ingress(
                group_id=security_group['GroupId'],
                ip_permissions=ip_permissions, ec2_client=ec2_client

            )

    else:
        vpc_id = ec2_client.describe_vpcs()['Vpcs'][0]['VpcId']

        security_group = aws_backoff.create_security_group(
            description='Security group for allowing SSH access to partitioning VM for Coysu Divina',
            group_name='divina-ssh',
            vpc_id=vpc_id, ec2_client=ec2_client
        )
        aws_backoff.authorize_security_group_ingress(
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
        waiter = aws_backoff.get_ec2_waiter('instance_running', ec2_client=ec2_client)
        waiter.wait(InstanceIds=[i['InstanceId'] for i in instance['Instances']])
        response = aws_backoff.describe_instances(instance_ids=[i['InstanceId'] for i in instance['Instances']],
                                      ec2_client=ec2_client)
        instance = response['Reservations'][0]['Instances'][0]

    except Exception as e:
        if not keep_instances_alive:
            aws_backoff.stop_instances(instance_ids=[instance['InstanceId']], ec2_client=ec2_client)
        raise e
    return instance, paramiko_key


def build_dataset_ssh(instance, verbosity, paramiko_key, divina_pip_arguments):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_ssh(client, hostname=instance['PublicIpAddress'], username="ec2-user", pkey=paramiko_key)
    commands = ['sudo cp /var/lib/cloud/instance/user-data.txt /home/ec2-user/user-data.json',
                'sudo yum install unzip -y', 'sudo yum install python3 -y', 'sudo yum install gcc -y',
                'sudo curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"',
                'sudo unzip awscliv2.zip',
                'sudo ./aws/install -i /usr/local/aws-cli -b /usr/local/bin',
                'sudo python3 -m pip install divina[dataset]=={} {}'.format(
                    get_distribution('divina').version, "" if divina_pip_arguments is None else divina_pip_arguments),
                'aws s3 cp s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/data_definition.json /home/ec2-user/data_definition.json'.format(
                    os.environ['VISION_ID']),
                'sudo chown -R ec2-user /home/ec2-user',
                'divina build-dataset']
    for cmd in commands:
        stdin, stdout, stderr = client.exec_command(cmd)
        if verbosity > 0:
            for line in stdout:
                sys.stdout.write(line)
        exit_status = stdout.channel.recv_exit_status()
        if not exit_status == 0:
            if verbosity > 2:
                for line in stderr:
                    sys.stderr.write(line)
            client.close()
            return False
    client.close()
    return True


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
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config/emr_config.json'), 'r') as f:
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
    steps = aws_backoff.list_steps(emr_client, cluster['JobFlowId'])
    last_step_id = steps['Steps'][0]['Id']
    emr_waiter = aws_backoff.get_emr_waiter(emr_client, 'step_complete')
    emr_waiter.wait(
        ClusterId=cluster['JobFlowId'],
        StepId=last_step_id,
        WaiterConfig={
            "Delay": 30,
            "MaxAttempts": 120
        }
    )
    return cluster


def run_command_emr(emr_client, cluster_id, keep_instances_alive, args):
    if keep_instances_alive:
        on_failure = 'CANCEL_AND_WAIT'
    else:
        on_failure = 'TERMINATE_CLUSTER'
    steps = emr_client.add_job_flow_steps(
        JobFlowId=cluster_id,
        ###TODO start here and translate this into using the cli somehow
        Steps=[
            {
                "Name": "vision_script",
                "ActionOnFailure": on_failure,
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": args
                }
            }
        ]
    )
    emr_waiter = aws_backoff.get_emr_waiter(emr_client, 'step_complete')
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
        response = aws_backoff.get_products(pricing_client, products_params)
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


def create_vision(import_bucket, divina_version, source_role=None, vision_role=None,
                  worker_profile='EMR_EC2_DefaultRole',
                  driver_role='EMR_DefaultRole', region='us-east-2', ec2_keyfile=None, data_definition=None,
                  keep_instances_alive=False, verbosity=0, divina_pip_arguments=None):
    os.environ['VISION_ID'] = str(round(datetime.datetime.now().timestamp()))
    os.environ['DIVINA_BUCKET'] = 'coysu-divina-prototype-visions'
    os.environ['IMPORT_BUCKET'] = import_bucket

    sys.stdout.write('Authenticating to the cloud...\n')
    source_session = boto3.session.Session(aws_access_key_id=os.environ['SOURCE_AWS_ACCESS_KEY_ID'],
                                           aws_secret_access_key=os.environ['SOURCE_AWS_SECRET_ACCESS_KEY'],
                                           region_name=region)

    vision_session = boto3.session.Session(aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                                           aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                                           region_name=region)

    ###current scope is that all supplied files are either a single endogenous schema OR an exogenous signal that can be joined to the endogenous schema
    vision_setup(vision_session=vision_session, source_session=source_session,
                 vision_role=vision_role, source_role=source_role, worker_profile=worker_profile,
                 driver_role=driver_role, vision_role_name='divina-vision-role',
                 ec2_keyfile=ec2_keyfile,
                 data_definition=data_definition, keep_instances_alive=keep_instances_alive, verbosity=verbosity,
                 divina_version=divina_version, divina_pip_arguments=divina_pip_arguments)

@backoff.on_exception(backoff.expo,
                      paramiko.client.NoValidConnectionsError)
def connect_ssh(client, hostname, username, pkey):
    client.connect(hostname=hostname, username=username, pkey=pkey)
    return client