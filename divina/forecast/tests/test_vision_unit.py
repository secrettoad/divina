import os
import json
from unittest.mock import patch
from divina.forecast.vision import import_data, create_dataset_ec2, create_modelling_emr, get_sessions
import boto3
import paramiko


def upload_s3_data(s3_client, df, key, region, bucket_name):
    s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={
        'LocationConstraint': region
    })
    df.to_csv('/tmp/test_df_1.csv')
    with open('/tmp/test_df_1.csv', 'w+') as f:
        s3_client.put_object(Bucket=bucket_name, Key=key, Body=f.read())


def test_session_creation(monkeypatch, vision_iam, source_iam, vision_sts, source_sts, environment):
    monkeypatch.setenv('IMPORT_BUCKET', 'source-bucket')
    get_sessions(vision_iam=vision_iam, source_iam=source_iam, vision_sts=vision_sts, source_sts=source_sts)
    assert(all([x in [y['RoleName'] for y in vision_iam.list_roles()['Roles']] for x in ['divina-source-role', 'divina-vision-role']]))


def test_data_import(monkeypatch, test_df_1, source_s3, vision_s3, environment):
    upload_s3_data(source_s3, test_df_1, 'test_df_1.csv', 'us-west-2', 'source-bucket')
    monkeypatch.setenv('IMPORT_BUCKET', 'source-bucket')
    import_data(vision_s3_client=vision_s3, source_s3_client=source_s3,
                vision_role_name='test_role')
    vision_files = vision_s3.list_objects(Bucket='coysu-divina-prototype-visions',
                                          Prefix='coysu-divina-prototype-{}/data/'.format(os.environ['VISION_ID']))
    source_files = source_s3.list_objects(Bucket='source-bucket')
    assert (set([k['Key'].split('/')[-1] for k in vision_files['Contents']]) == set(
        [k['Key'].split('/')[-1] for k in source_files['Contents']]))


def test_validate_data_definition():
    ###TODO implement this
    pass


def test_dataset_infrastructure_df1(monkeypatch, test_df_1, vision_s3, vision_ec2, divina_test_version, environment):
    upload_s3_data(vision_s3, test_df_1, 'test_df_1.csv', 'us-west-2', 'divina-bucket')
    monkeypatch.setenv('DIVINA_BUCKET', 'divina-bucket')
    with patch('boto3.client'):
        mock_vision_pricing = boto3.client('pricing')
    with open(os.path.join('../mocks', 'pricing_mock.json'), 'r') as f:
        mock_vision_pricing.get_products.return_value = json.load(f)
    instance, paramiko_key = create_dataset_ec2(s3_client=vision_s3, ec2_client=vision_ec2, pricing_client=mock_vision_pricing, divina_version=divina_test_version)
    assert (all([x in instance for x in ['ImageId', 'InstanceId']]) and instance['State'] == 'running' and type(paramiko_key) == paramiko.rsakey.RSAKey)


def test_dataset_build_df1(test_df_1):
    ##TODO test this with docker
    pass


def test_modelling_cluster_creation(vision_emr):
    cluster = create_modelling_emr(vision_emr)
    assert (all([x in cluster for x in ['JobFlowId', 'ClusterArn']]))


def test_train_output1(test_df_1, spark_context):
    ###TODO implement with kubernetes
    pass


def test_predict_output1():
    ###TODO implement with kubernetes
    pass


def test_validate_output1():
    ###TODO implement with kubernetes
    pass

