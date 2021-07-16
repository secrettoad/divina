import pytest
import boto3
import moto
import os

from divina.vision import import_data


def upload_s3_data(s3_client, df, region):
    s3_client.create_bucket(Bucket='source-bucket', CreateBucketConfiguration={
                          'LocationConstraint': region
                      })
    df.to_csv('/tmp/test_df_1.csv')
    with open('/tmp/test_df_1.csv', 'w+') as f:
        s3_client.put_object(Bucket='source-bucket', Key='test_df_1.csv', Body=f.read())


def test_role_creation():
    pass


def test_data_import(monkeypatch, test_df_1, source_s3, vision_s3, environment):
    upload_s3_data(source_s3, test_df_1, 'us-west-2')
    import_data(vision_s3_client=vision_s3, source_s3_client=source_s3, source_bucket='source-bucket', vision_role_name='test_role', vision_region='us-east-2')
    vision_files = vision_s3.list_objects(Bucket='coysu-divina-prototype-visions', Prefix='coysu-divina-prototype-{}/data/'.format(os.environ['VISION_ID']))
    source_files = source_s3.list_objects(Bucket='source-bucket')
    assert(set([k['Key'].split('/')[-1] for k in vision_files['Contents']]) == set([k['Key'].split('/')[-1] for k in source_files['Contents']]))


def test_dataset_build_data_definition():
    pass


def test_dataset_build_output1():
    pass


def test_dataset_build_partition_sizes():
    pass


def test_train_glasma_output1():
    pass


def test_predict_glasma_output1():
    pass


def test_validate_glasma_output1():
    pass