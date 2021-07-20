import os
import json
from unittest.mock import patch
from divina.vision.vision import import_data, build_dataset_infrastructure, upload_scripts, run_training
import boto3
import paramiko
from pyspark.sql import SQLContext


def upload_s3_data(s3_client, df, key, region):
    s3_client.create_bucket(Bucket='source-bucket', CreateBucketConfiguration={
        'LocationConstraint': region
    })
    df.to_csv('/tmp/test_df_1.csv')
    with open('/tmp/test_df_1.csv', 'w+') as f:
        s3_client.put_object(Bucket='source-bucket', Key=key, Body=f.read())


def test_role_creation():
    pass


def test_data_import(test_df_1, source_s3, vision_s3, environment):
    upload_s3_data(source_s3, test_df_1, 'test_df_1.csv', 'us-west-2')
    import_data(vision_s3_client=vision_s3, source_s3_client=source_s3, source_bucket='source-bucket',
                vision_role_name='test_role', vision_region='us-east-2')
    vision_files = vision_s3.list_objects(Bucket='coysu-divina-prototype-visions',
                                          Prefix='coysu-divina-prototype-{}/data/'.format(os.environ['VISION_ID']))
    source_files = source_s3.list_objects(Bucket='source-bucket')
    assert (set([k['Key'].split('/')[-1] for k in vision_files['Contents']]) == set(
        [k['Key'].split('/')[-1] for k in source_files['Contents']]))


def test_dataset_validate_data_definition():
    ###TODO implement this
    pass


def test_dataset_infrastructure_output1(monkeypatch, test_df_1, vision_s3, vision_ec2, environment):
    upload_s3_data(vision_s3, test_df_1, 'test_df_1.csv', 'us-west-2')
    upload_scripts(vision_s3, 'source-bucket')
    with patch('boto3.client'):
        mock_vision_pricing = boto3.client('pricing')
    with open('pricing_mock.json', 'r') as f:
        mock_vision_pricing.get_products.return_value = json.load(f)
    instance, paramiko_key = build_dataset_infrastructure(s3_client=vision_s3, ec2_client=vision_ec2, pricing_client=mock_vision_pricing, data_bucket='source-bucket')
    assert (all([x in instance for x in ['ImageId', 'InstanceId']]) and instance['State'] == 'running' and type(paramiko_key) == paramiko.rsakey.RSAKey)


def test_dataset_build_output1(spark_context, test_df_1):
    def run_training(emr_client, worker_profile='EMR_EC2_DefaultRole',
                                  driver_role='EMR_DefaultRole', keep_instances_alive=False, ec2_key=None):
        if keep_instances_alive:
            on_failure = 'CANCEL_AND_WAIT'
        else:
            on_failure = 'TERMINATE_CLUSTER'

        with open(os.path.join('..', 'divina/config/emr_config.json'), 'r') as f:
            emr_config = json.loads(
                os.path.expandvars(json.dumps(json.load(f)).replace('${WORKER_PROFILE}', worker_profile).replace(
                    '${DRIVER_ROLE}',
                    driver_role).replace(
                    '${EMR_ON_FAILURE}',
                    on_failure).replace(
                    '${EC2_KEYNAME}', ec2_key)))
        emr_config['emr_config']['Instances']['KeepJobFlowAliveWhenNoSteps'] = keep_instances_alive
        cluster = emr_client.run_job_flow(**emr_config['emr_config'])
        return cluster

    df = mf.transform(df)
    assert (type(df) == DataFrame and df.columns == ['a', 'b', 'c'])
    pass


def test_train_output1():

    pass


def test_predict_output1():
    pass


def test_validate_output1():
    pass
