import os
import sys
import pandas as pd
import numpy as np
import json
from zipfile import ZipFile
from io import BytesIO
import traceback
import shutil
from .errors import FileTypeNotSupported
import pathlib
from pkg_resources import get_distribution
import math
import paramiko
from ..aws import aws_backoff
from ..aws.utils import unnest_ec2_price, ec2_pricing
import io
import dask.dataframe as dd


def partition_data(dataset_directory, dataset_id, files, partition_dimensions=None):
    for file in files:
        partition_size = 5000000000
        memory_usage = file['df'].memory_usage(deep=True).sum()
        path = os.path.join(dataset_directory,
                                 dataset_id, 'data')
        if int(memory_usage) <= partition_size:
            dd.from_pandas(file['df'], chunksize=10000).to_parquet(path, write_index=False)
        elif partition_dimensions:
            dd.from_pandas(file['df'], chunksize=10000).to_parquet(
                path, write_index=False, partition_on=partition_dimensions)
        else:
            partition_rows = partition_size / memory_usage * len(file['df'])
            file['df']['partition'] = np.arange(len(file['df'])) // partition_rows
            partition_cols = ['partition']
            dd.from_pandas(file['df'], chunksize=10000).to_parquet(
                path, write_index=False, partition_on=partition_cols)
        sys.stdout.write('SAVED PARQUET - {} - {}\n'.format(dataset_id, file['source_path']))


def profile_data(dataset_directory, dataset_id, files):
    for file in files:
        path = os.path.join(dataset_directory,
                                 dataset_id, 'profile')
        dd.from_pandas(file['df'].describe().reset_index(), chunksize=10000).to_parquet(os.path.join(path), write_index=False)


def decompress_file(tmp_dir, key, s3, data_directory):
    data = s3.open(
        os.path.join('{}'.format(data_directory), key.split('/')[-1])).read()
    files = []
    if key.split('.')[-1] == 'zip':
        if not os.path.isdir('{}/{}'.format(tmp_dir, key.split('/')[-1].replace('.', '-'))):
            os.mkdir('{}/{}'.format(tmp_dir, key.split('/')[-1].replace('.', '-')))
        zip_file = ZipFile(BytesIO(data))
        for name in zip_file.namelist():
            local_path = '{}/{}'.format(tmp_dir, os.path.join(key.split('/')[-1].replace('.', '-'), name))
            with open(local_path, 'wb+') as f:
                f.write(zip_file.read(name))
            files.append({'source_path': os.path.join(key, name), 'local_path': local_path,
                          'filename': local_path.split('/')[-1]})
    else:
        local_path = '{}/{}'.format(tmp_dir, key.split('/')[-1])
        if type(data) == bytes:
            with open(local_path, 'wb+') as f:
                f.write(data)
        elif type(data) == str:
            with open(local_path, 'w+') as f:
                f.write(data)
        else:
            raise Exception('file {} is not string or bytes. cannot write locally'.format(key))
        files = [{'source_path': key, 'local_path': local_path,
                  'filename': local_path.split('/')[-1]}]
    return files


def parse_files(files):
    for file in files:
        try:
            extension = file['local_path'].split('.')[-1]
            if extension == 'csv':
                df = pd.read_csv(file['local_path'])
            else:
                raise FileTypeNotSupported(extension)
            file['df'] = df
        except Exception as e:
            traceback.print_exc()
            raise e

    return files


def rm(f):
    if os.path.isdir(f): return os.rmdir(f)
    if os.path.isfile(f): return os.unlink(f)
    raise TypeError('must be either file or directory')


def build_dataset(s3_fs, dataset_directory, data_directory, dataset_id, tmp_dir='tmp_data', partition_dimensions=None):
    if os.path.exists('./environment.json'):
        with open('./environment.json') as f:
            os.environ.update(json.load(f)['ENVIRONMENT'])
    data_directories = ['data', 'profile']
    pathlib.Path(tmp_dir).mkdir(
        parents=True, exist_ok=True)
    for d in data_directories:
        pathlib.Path(os.path.join('{}/{}'.format(dataset_directory, dataset_id),
                                  d)).mkdir(
            parents=True, exist_ok=True)
    for key in s3_fs.ls(
            '{}'.format(data_directory)):
        try:
            files = decompress_file(tmp_dir=tmp_dir, key=key, s3=s3_fs,
                                    data_directory=data_directory)
            files = parse_files(files=files)
            partition_data(dataset_directory=dataset_directory, dataset_id=dataset_id, files=files, partition_dimensions=partition_dimensions)
            profile_data(dataset_directory=dataset_directory, dataset_id=dataset_id, files=files)
        except Exception as e:
            sys.stdout.write('Could not partition file: {}\n'.format(key))
            traceback.print_exc()
            shutil.rmtree('{}'.format(tmp_dir))
            raise e
    shutil.rmtree('{}'.format(tmp_dir))


def create_partitioning_ec2(s3_fs, data_directory, vision_session, ec2_keyfile=None,
                       keep_instances_alive=False):

    ec2_client = vision_session.client('ec2')
    pricing_client = vision_session.client('pricing', region_name='us-east-1')

    file_sizes = []
    keys = []

    for o in s3_fs.ls(data_directory):
        file_sizes.append(
            s3_fs.info(o)['size'])
        keys.append(o)

    ec2_keys = ec2_client.describe_key_pairs()
    if 'divina_ec2_key' in [x['KeyName'] for x in ec2_keys['KeyPairs']]:
        ec2_client.delete_key_pair(KeyName='divina_ec2_key')
    if ec2_keyfile:
        with open(os.path.join(os.path.expanduser('~'), '.ssh', ec2_keyfile + '.pub')) as f:
            key = aws_backoff.import_key_pair(
                key_name='divina_ec2_key',
                public_key_material=f.read(), ec2_client=ec2_client
            )
            paramiko_key = paramiko.RSAKey.from_private_key_file(
                os.path.join(os.path.expanduser('~'), '.ssh', ec2_keyfile), password=os.environ['KEY_PASS'])
    else:
        key = ec2_client.create_key_pair(
            KeyName='divina_ec2_key'
        )
        paramiko_key = paramiko.RSAKey.from_private_key(io.StringIO(key['KeyMaterial']))

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


def build_dataset_ssh(instance, verbosity, paramiko_key, dataset_directory, dataset_id, divina_pip_arguments):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    paramiko.connect_ssh(client, hostname=instance['PublicIpAddress'], username="ec2-user", pkey=paramiko_key)
    commands = ['sudo echo \'{}\' > /home/ec2-user/environment.json'.format(json.dumps({'ENVIRONMENT': {'DATASET_ID': dataset_id,
                        'DATASET_BUCKET': dataset_directory}})),
                'sudo yum install unzip -y', 'sudo yum install python3 -y', 'sudo yum install gcc -y',
                'sudo python3 -m pip install divina[dataset]=={} {}'.format(
                    get_distribution('divina').version, "" if divina_pip_arguments is None else divina_pip_arguments),
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


def get_dataset(vision_definition):

    df = dd.read_parquet("{}/{}/data/*".format(vision_definition['dataset_directory'],
                                               vision_definition['dataset_id']))
    profile = dd.read_parquet("{}/{}/profile/*".format(vision_definition['dataset_directory'],
                                               vision_definition['dataset_id']))
    if 'joins' in vision_definition:
        for i, join in enumerate(vision_definition['joins']):

            join_df = dd.read_parquet("{}/{}/data/*".format(join['dataset_directory'],
                                               join['dataset_id']))
            join_df.columns = ['{}_{}'.format(join['dataset_id'], c) for c in join_df.columns]
            df = df.merge(join_df, how='left', left_on=join['join_on'][0], right_on=join['join_on'][1])

            join_profile = dd.read_parquet("{}/{}/profile/*".format(join['dataset_directory'],
                                                            join['dataset_id']))
            join_profile.columns = ['{}_{}'.format(join['dataset_id'], c) for c in join_profile.columns]
            profile = dd.concat([profile, join_profile], axis=1)

    return df, profile
