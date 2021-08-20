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
import math
from .aws import aws_backoff
from .aws.utils import unnest_ec2_price, ec2_pricing
from botocore.exceptions import WaiterError
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
        dd.from_pandas(file['df'].describe().reset_index(), chunksize=10000).to_parquet(os.path.join(path),
                                                                                        write_index=False)


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
            partition_data(dataset_directory=dataset_directory, dataset_id=dataset_id, files=files,
                           partition_dimensions=partition_dimensions)
            profile_data(dataset_directory=dataset_directory, dataset_id=dataset_id, files=files)
        except Exception as e:
            sys.stdout.write('Could not partition file: {}\n'.format(key))
            traceback.print_exc()
            shutil.rmtree('{}'.format(tmp_dir))
            raise e
    shutil.rmtree('{}'.format(tmp_dir))


def build_remote(commit, s3_fs, read_path, write_path, dataset_name, ec2_client, pricing_client, ec2_keypair_name=None,
           keep_instances_alive=False, partition_dimensions=None):
    file_sizes = []
    keys = []
    if read_path[:5] == 's3://':
        for o in s3_fs.ls(read_path):
            file_sizes.append(
                s3_fs.info(o)['size'])
            keys.append(o)
    else:
        for (dirpaths, dirnames, filenames) in os.walk(read_path):
            for filename in filenames:
                file_sizes.append(
                    os.stat(pathlib.Path(read_path, filename)).st_size)
                keys.append(filename)

    if len(keys) < 1:
        raise Exception('No files found at read_path: {}:'.format(read_path))

    if ec2_keypair_name:
        ec2_keys = ec2_client.describe_key_pairs()
        if ec2_keypair_name not in [x['KeyName'] for x in ec2_keys['KeyPairs']]:
            raise Exception('EC2 keypair {} not found'.format(ec2_keypair_name))
        else:
            key = [x for x in ec2_keys['KeyPairs'] if x['KeyName'] == ec2_keypair_name][0]

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

    ###TODO start here test --local locally
    userdata = '''#!/bin/bash
                       ( ( sudo yum install python3 -y;
                       sudo yum install git -y;
                       sudo python3 -m pip install git+https://git@github.com/secrettoad/divina.git@{};
                       divina dataset build {} {} {} --local) && sudo shutdown now; ) || sudo shutdown now -h;'''.format(
        commit, read_path, write_path, dataset_name,
    )

    if not ec2_keypair_name:
        instance = ec2_client.run_instances(ImageId='ami-0b223f209b6d4a220', MinCount=1, MaxCount=1,
                                            IamInstanceProfile={'Name': 'EMR_EC2_DefaultRole'},
                                            InstanceType=partitioning_instance_type['product']['attributes'][
                                                'instanceType'],
                                            BlockDeviceMappings=[
                                                {"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": required_disk}}],
                                            UserData=userdata
                                            )['Instances'][0]
    else:
        instance = ec2_client.run_instances(ImageId='ami-0b223f209b6d4a220', MinCount=1, MaxCount=1,
                                            IamInstanceProfile={'Name': 'EMR_EC2_DefaultRole'},
                                            InstanceType=partitioning_instance_type['product']['attributes'][
                                                'instanceType'],
                                            KeyName=key['KeyName'],
                                            BlockDeviceMappings=[
                                                {"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": required_disk}}],
                                            UserData=userdata
                                            )['Instances'][0]

    try:
        running_waiter = aws_backoff.get_ec2_waiter('instance_running', ec2_client=ec2_client)
        running_waiter.wait(InstanceIds=[instance['InstanceId']])
        stopped_waiter = aws_backoff.get_ec2_waiter('instance_stopped', ec2_client=ec2_client)
        stopped_waiter.wait(InstanceIds=[instance['InstanceId']])
        response = aws_backoff.describe_instances(instance_ids=[instance['InstanceId']],
                                                  ec2_client=ec2_client)
        instance = response['Reservations'][0]['Instances'][0]
        if not keep_instances_alive:
            aws_backoff.stop_instances(instance_ids=[instance['InstanceId']], ec2_client=ec2_client)
    except WaiterError as e:
        instance = e.last_response['Reservations'][0]['Instances'][0]
        if instance['State']['Name'] == 'shutting-down':
            ###TODO implement EC2 logging
            raise Exception('Remote error during dataset build. Instance terminated. Log file can be found here: ')
        else:
            raise e
    finally:
        if not keep_instances_alive:
            aws_backoff.stop_instances(instance_ids=[instance['InstanceId']], ec2_client=ec2_client)
    return instance


def _build(commit, s3_fs, read_path, write_path, dataset_name, ec2_client, pricing_client, ec2_keypair_name=None,
           keep_instances_alive=False, local=False, partition_dimensions=None):

    if not local:
        build_remote(commit=commit, s3_fs=s3_fs, read_path=read_path, write_path=write_path, dataset_name=dataset_name, ec2_client=ec2_client, pricing_client=pricing_client, ec2_keypair_name=ec2_keypair_name,
           keep_instances_alive=keep_instances_alive, partition_dimensions=partition_dimensions)

    else:
        build_dataset(s3_fs=s3_fs, dataset_directory=write_path, data_directory=read_path, dataset_id=dataset_name, partition_dimensions=partition_dimensions)


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
