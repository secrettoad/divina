import boto3
import os
import datetime
import subprocess

import paramiko
import io
import sys
import json
from ..aws import aws_backoff
from .import_data import import_data
import backoff
from .errors import InvalidDataDefinitionException
from ..forecast.dataset import create_partitioning_ec2, build_dataset_ssh


####TODO abtract rootish from role jsons - use os.path.expandvars

def vision_setup(divina_version, worker_profile, driver_role, vision_session, source_session,
                 ec2_keyfile,
                 vision_role_name,
                 data_definition=None, keep_instances_alive=False, verbosity=0, vision_role=None, source_role=None, divina_pip_arguments=None):
    vision_iam = vision_session.client('iam')
    vision_sts = vision_session.client('sts')
    vision_session = get_vision_session(vision_iam=vision_iam,
                                                  vision_role=vision_role,
                                                  vision_sts=vision_sts
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
    instance, paramiko_key = create_partitioning_ec2(ec2_client=vision_ec2_client,
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


def validate_vision_definition(vision_definition):
    if not 'time_index' in vision_definition:
        raise InvalidDataDefinitionException('required field time_index not found in data definition')
    if not 'target' in vision_definition:
        raise InvalidDataDefinitionException('required field target not found in data definition')
    if "time_validation_splits" in vision_definition:
        if not type(vision_definition['time_validation_splits']) == list:
            raise InvalidDataDefinitionException('time_validation_splits must be a list of date-like strings')
        elif not all([type(x) == str for x in vision_definition['time_validation_splits']]):
            raise InvalidDataDefinitionException('time_validation_splits must be a list of date-like strings')
    if "time_horizons" in vision_definition:
        if not type(vision_definition['time_horizons']) == list:
            raise InvalidDataDefinitionException('time_horizons must be a list of integers')
        elif not all([type(x) == int for x in vision_definition['time_horizons']]):
            raise InvalidDataDefinitionException('time_horizons must be a list of integers')


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
