import boto3
import os
import datetime
import subprocess
import paramiko
import sys
import json
from divina.divina.aws import aws_backoff
import backoff
from .errors import InvalidDataDefinitionException
import pathlib


####TODO abtract rootish from role jsons - use os.path.expandvars


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


def create_vision(s3_fs,
                  worker_profile='EMR_EC2_DefaultRole',
                  driver_role='EMR_DefaultRole', region='us-east-2', ec2_keyfile=None, vision_definition=None,
                  keep_instances_alive=False, verbosity=0, divina_directory, divina_pip_arguments=None):
    os.environ['VISION_ID'] = str(round(datetime.datetime.now().timestamp()))
    os.environ['DIVINA_BUCKET'] = 'coysu-divina-prototype-visions'

    sys.stdout.write('Authenticating to the cloud...\n')
    try:
        vision_session = boto3.session.Session(profile_name='divina', region_name=region)
    except:
        raise Exception('No AWS profile named "divina" found')

    if not worker_profile and driver_role:
        sys.stdout.write('Creating spark driver and executor cloud roles...\n')
        create_emr_roles(vision_session)

    vision_session = boto3.Session()

    with s3_fs.open(pathlib.Path(divina_directory, '{}/vision_definition.json'.format(os.environ['VISION_ID'])), 'w+') as f:
        json.dump(vision_definition, f)

    sys.stdout.write('Building dataset...\n')

    sys.stdout.write('Creating forecasts...\n')
    emr_client = vision_session.client('emr')
    emr_cluster = create_modelling_emr(emr_client=emr_client,
                                       worker_profile=worker_profile, driver_role=driver_role,
                                       keep_instances_alive=keep_instances_alive,
                                       ec2_key='vision_{}_ec2_key'.format(os.environ['VISION_ID']))

    run_command_emr(emr_client=emr_client, cluster_id=emr_cluster['JobFlowId'],
                    keep_instances_alive=keep_instances_alive,
                    args=['divina', 'train', '--data_definition', '/home/hadoop/data_definition.json', ''])
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