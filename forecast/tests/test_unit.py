import os
import json
from unittest.mock import patch, MagicMock
from ..import_data import import_data
from ..vision import create_partitioning_ec2, create_modelling_emr, get_sessions, validate_vision_definition
from ..train import dask_train
from ..predict import dask_predict
from ..dataset import build_dataset, get_dataset
from ..validate import dask_validate
from ..errors import InvalidDataDefinitionException
import boto3
import paramiko
import io
import shutil
import pathlib
import pytest
from ...models.dask.ensembles.linear import GLASMA
import joblib
import pandas as pd
import dask.dataframe as ddf



def test_session_creation(monkeypatch, vision_iam, source_iam, vision_sts, source_sts, environment):
    get_sessions(vision_iam=vision_iam, source_iam=source_iam, vision_sts=vision_sts, source_sts=source_sts)
    assert (all([x in [y['RoleName'] for y in vision_iam.list_roles()['Roles']] for x in
                 ['divina-source-role', 'divina-vision-role']]))


@patch.dict(os.environ, {"VISION_ID": "test1"})
def test_data_import(test_df_1, source_s3, vision_s3, environment):
    source_s3.create_bucket(Bucket=os.environ['IMPORT_BUCKET'], CreateBucketConfiguration={
        'LocationConstraint': 'us-west-2'
    })
    bytesIO = io.BytesIO()
    test_df_1.to_csv(bytesIO)
    source_s3.put_object(Bucket=os.environ['IMPORT_BUCKET'], Key='test_df_1.csv', Body=bytesIO)
    import_data(vision_s3_client=vision_s3, source_s3_client=source_s3,
                vision_role_name='test_role')
    vision_files = vision_s3.list_objects(Bucket='coysu-divina-prototype-visions',
                                          Prefix='coysu-divina-prototype-{}/data/'.format(os.environ['VISION_ID']))
    source_files = source_s3.list_objects(Bucket='source-bucket')
    assert (set([k['Key'].split('/')[-1] for k in vision_files['Contents']]) == set(
        [k['Key'].split('/')[-1] for k in source_files['Contents']]))


def test_validate_vision_definition(vd_no_target, vd_time_horizons_not_list, vd_time_validation_splits_not_list,
                                    vd_no_time_index, vd_no_dataset_id, vd_no_dataset_directory):
    for dd in [vd_no_target, vd_no_time_index, vd_time_validation_splits_not_list, vd_time_horizons_not_list,
               vd_no_dataset_id, vd_no_dataset_directory]:
        try:
            validate_vision_definition(dd)
        except InvalidDataDefinitionException:
            assert True
        else:
            assert False


@patch('s3fs.S3FileSystem.open', open)
@patch('s3fs.S3FileSystem.ls', os.listdir)
@patch.dict(os.environ, {"VISION_ID": "test1"})
def test_dataset_infrastructure_df1(s3_fs, test_df_1, vision_s3, vision_ec2, divina_test_version, ec2_pricing_stub,
                                    environment):
    test_df_1.to_csv(
        os.path.join(os.environ['IMPORT_BUCKET'], 'test_df_1.csv'), index=False)
    with patch('boto3.client'):
        mock_vision_pricing = boto3.client('pricing')
    mock_vision_pricing.get_products.return_value = ec2_pricing_stub
    instance, paramiko_key = create_partitioning_ec2(s3_fs=s3_fs, ec2_client=vision_ec2,
                                                pricing_client=mock_vision_pricing, divina_version=divina_test_version,
                                                data_directory=os.environ['IMPORT_BUCKET'])
    assert (all([x in instance for x in ['ImageId', 'InstanceId']]) and instance['State']['Name'] == 'running' and type(
        paramiko_key) == paramiko.rsakey.RSAKey)


@patch('s3fs.S3FileSystem.open', open)
@patch('s3fs.S3FileSystem.ls', os.listdir)
def test_dataset_build(vision_s3, test_df_1, environment):
    try:
        pathlib.Path(
            os.environ['IMPORT_BUCKET']).mkdir(
            parents=True, exist_ok=True)
        test_df_1.to_csv(
            os.path.join(os.environ['IMPORT_BUCKET'], 'test_df_1.csv'), index=False)
        pathlib.Path(
            os.environ['DATASET_BUCKET'], os.environ['DATASET_ID']).mkdir(
            parents=True, exist_ok=True)
        build_dataset(dataset_directory=os.environ['DATASET_BUCKET'], data_directory=os.environ['IMPORT_BUCKET'],
                      dataset_id=os.environ['DATASET_ID'])

        '''Uncomment below to generate stub models'''
        pathlib.Path(
            os.path.join('stubs',
                         '{}'.format(os.environ['DATASET_ID']))).mkdir(
            parents=True, exist_ok=True)

        shutil.copytree(
            os.path.join(os.environ['DATASET_BUCKET'], '{}'.format(os.environ['DATASET_ID'])), os.path.join('stubs',
                                                                                                            '{}'.format(
                                                                                                                os.environ[
                                                                                                                    'DATASET_ID'])),

            dirs_exist_ok=True)

        '''end stubbing'''

        pd.testing.assert_frame_equal(ddf.read_parquet(
            os.path.join(os.environ['DATASET_BUCKET'], os.environ['DATASET_ID'],
                         'data/*')).compute(), ddf.read_parquet(
            os.path.join('stubs', os.environ['DATASET_ID'],
                         'data/*')).compute())
        pd.testing.assert_frame_equal(ddf.read_parquet(
            os.path.join(os.environ['DATASET_BUCKET'], os.environ['DATASET_ID'], 'profile')).compute(),
                                      ddf.read_parquet(
                                          os.path.join('stubs', os.environ['DATASET_ID'], 'profile')).compute())
    finally:
        shutil.rmtree(os.environ['VISION_BUCKET'], ignore_errors=True)
        shutil.rmtree(os.environ['DATASET_BUCKET'], ignore_errors=True)


def test_modelling_cluster_creation(vision_emr):
    ###TODO implement kubernetes so that this is more unit testable
    mock_waiter = MagicMock(spec=["wait"])
    mock_waiter.wait = None
    vision_emr.get_waiter = mock_waiter
    cluster = create_modelling_emr(vision_emr)
    assert (all([x in cluster for x in ['JobFlowId', 'ClusterArn']]))


@patch('s3fs.S3FileSystem.open', open)
@patch('s3fs.S3FileSystem.ls', os.listdir)
@patch.dict(os.environ, {"VISION_ID": "test1"})
def test_dask_train(vision_s3, test_df_1, test_vd_1, dask_client, environment):
    try:
        pathlib.Path(
            os.path.join(os.environ['DATASET_BUCKET'], '{}'.format(os.environ['DATASET_ID']),
                         'data')).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(
            os.path.join(os.environ['DATASET_BUCKET'], '{}'.format(os.environ['DATASET_ID']),
                         'profile')).mkdir(
            parents=True, exist_ok=True)
        ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(os.path.join(os.environ['DATASET_BUCKET'],
                                                                            os.environ['DATASET_ID'], 'data'))
        ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(os.path.join(os.environ['DATASET_BUCKET'],
                                                                                       os.environ['DATASET_ID'],
                                                                                       'profile'))
        with open(os.path.join(os.environ['DATASET_BUCKET'], '{}'.format(os.environ['DATASET_ID']),
                               'vision_definition.json'), 'w+') as f:
            json.dump(test_vd_1, f)
        models = dask_train(dask_client=dask_client, dask_model=GLASMA,
                            vision_definition=test_vd_1['vision_definition'],
                            vision_id=os.environ['VISION_ID'], divina_directory=os.environ['VISION_BUCKET'])
        pathlib.Path(
            pathlib.Path(__file__).resolve().parent, 'stubs',
            'coysu-divina-prototype-{}'.format(os.environ['VISION_ID'])).mkdir(
            parents=True, exist_ok=True)

        '''Uncomment below to generate stub models'''

        pathlib.Path(pathlib.Path(pathlib.Path(__file__).resolve().parent, 'stubs',
                                  'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']), 'models')).mkdir(
            parents=True, exist_ok=True)
        for model in models:
            joblib.dump(models[model],
                        str(pathlib.Path(pathlib.Path(__file__).resolve().parent, 'stubs',
                                         'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']), 'models', model)))

        '''end stubbing'''

        assert (joblib.load(os.path.abspath(
            os.path.join(os.environ['VISION_BUCKET'], 'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']),
                         'models', 's-19700101-000008_h-1'))) == joblib.load(
            os.path.abspath(os.path.join('stubs', 'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']),
                                         'models', 's-19700101-000008_h-1'))))
    finally:
        shutil.rmtree(os.environ['VISION_BUCKET'], ignore_errors=True)
        shutil.rmtree(os.environ['DATASET_BUCKET'], ignore_errors=True)


@patch('s3fs.S3FileSystem.open', open)
@patch('s3fs.S3FileSystem.ls', os.listdir)
@patch.dict(os.environ, {"VISION_ID": "test2"})
def test_get_composite_dataset(vision_s3, test_df_1, test_df_2, test_vd_2,
                               test_composite_profile_1, test_composite_df_1, dask_client, environment):
    try:
        pathlib.Path(
            os.path.join(os.environ['DATASET_BUCKET'], '{}'.format(os.environ['DATASET_ID']),
                         'data')).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(
            os.path.join(os.environ['DATASET_BUCKET'], '{}'.format(os.environ['DATASET_ID']),
                         'profile')).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(
            os.path.join(os.environ['DATASET_BUCKET'], '{}'.format(os.environ['DATASET_ID2']),
                         'data')).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(
            os.path.join(os.environ['DATASET_BUCKET'], '{}'.format(os.environ['DATASET_ID2']),
                         'profile')).mkdir(
            parents=True, exist_ok=True)
        ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(os.path.join(os.environ['DATASET_BUCKET'],
                                                                            os.environ['DATASET_ID'], 'data'))
        ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(os.path.join(os.environ['DATASET_BUCKET'],
                                                                                       os.environ['DATASET_ID'],
                                                                                       'profile'))
        ddf.from_pandas(test_df_2, chunksize=10000).to_parquet(os.path.join(os.environ['DATASET_BUCKET'],
                                                                            os.environ['DATASET_ID2'], 'data'))
        ddf.from_pandas(test_df_2.describe(), chunksize=10000).to_parquet(os.path.join(os.environ['DATASET_BUCKET'],
                                                                                       os.environ['DATASET_ID2'],
                                                                                       'profile'))
        df, profile = get_dataset(test_vd_2['vision_definition'])

        pd.testing.assert_frame_equal(df.compute(), test_composite_df_1)
        pd.testing.assert_frame_equal(profile.compute(), test_composite_profile_1)
    finally:
        shutil.rmtree(os.environ['VISION_BUCKET'], ignore_errors=True)
        shutil.rmtree(os.environ['DATASET_BUCKET'], ignore_errors=True)


@patch('s3fs.S3FileSystem.open', open)
@patch('s3fs.S3FileSystem.ls', os.listdir)
@patch.dict(os.environ, {"VISION_ID": "test1"})
def test_dask_predict_output1(dask_client, test_df_1, test_vd_1, test_model_1, environment):
    try:
        pathlib.Path(
            os.path.join(os.environ['DATASET_BUCKET'], os.environ['DATASET_ID'], 'data')).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(
            os.path.join(os.environ['VISION_BUCKET'], 'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']),
                         'models')).mkdir(
            parents=True, exist_ok=True)
        ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(os.path.join(os.environ['DATASET_BUCKET'],
                                          os.environ['DATASET_ID'], 'data'))
        ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(os.path.join(os.environ['DATASET_BUCKET'],
                                                                                       os.environ['DATASET_ID'],
                                                                                       'profile'))
        shutil.copy2(os.path.join('stubs', 'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']),
                                  'models', 's-19700101-000008_h-1'), os.path.join(os.environ['VISION_BUCKET'],
                                                                                   'coysu-divina-prototype-{}'.format(
                                                                                       os.environ['VISION_ID']),
                                                                                   'models', 's-19700101-000008_h-1'))
        with open(os.path.join(os.environ['VISION_BUCKET'], 'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']),
                               'vision_definition.json'), 'w+') as f:
            json.dump(test_vd_1, f)

        dask_predict(dask_client=dask_client, vision_definition=test_vd_1['vision_definition'],
                     vision_id=os.environ['VISION_ID'],
                     divina_directory=os.environ['VISION_BUCKET'])

        '''Uncomment below to generate stub models'''

        pathlib.Path(pathlib.Path(pathlib.Path(__file__).resolve().parent, 'stubs',
                                  'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']), 'predictions')).mkdir(
            parents=True, exist_ok=True)
        shutil.copytree(
            os.path.join(os.environ['VISION_BUCKET'], 'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']),
                         'predictions', 's-19700101-000008'), os.path.join('stubs',
                                                                           'coysu-divina-prototype-{}'.format(
                                                                               os.environ['VISION_ID']),
                                                                           'predictions', 's-19700101-000008'),
            dirs_exist_ok=True)

        '''end stubbing'''

        pd.testing.assert_frame_equal(ddf.read_parquet(
            os.path.join(os.environ['VISION_BUCKET'], 'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']),
                         'predictions', 's-19700101-000008')).compute(), ddf.read_parquet(
            os.path.join('stubs', 'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']),
                         'predictions', 's-19700101-000008')).compute())
    finally:
        shutil.rmtree(os.environ['VISION_BUCKET'], ignore_errors=True)
        shutil.rmtree(os.environ['DATASET_BUCKET'], ignore_errors=True)


@patch('s3fs.S3FileSystem.open', open)
@patch('s3fs.S3FileSystem.ls', os.listdir)
@patch.dict(os.environ, {"VISION_ID": "test1"})
def test_dask_validate_output1(test_vd_1, test_df_1, test_metrics_1, environment, dask_client):
    try:
        pathlib.Path(
            os.path.join(os.environ['DATASET_BUCKET'], os.environ['DATASET_ID'], 'data')).mkdir(
            parents=True, exist_ok=True)
        ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(os.path.join(os.environ['DATASET_BUCKET'],
                                                                            os.environ['DATASET_ID'], 'data'))
        ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(os.path.join(os.environ['DATASET_BUCKET'],
                                                                                       os.environ['DATASET_ID'],
                                                                                       'profile'))
        pathlib.Path(
            os.path.join(os.environ['VISION_BUCKET'], 'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']),
                         'predictions')).mkdir(
            parents=True, exist_ok=True)

        shutil.copytree(os.path.join('stubs', 'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']),
                                     'predictions', 's-19700101-000008'), os.path.join(os.environ['VISION_BUCKET'],
                                                                                       'coysu-divina-prototype-{}'.format(
                                                                                           os.environ['VISION_ID']),
                                                                                       'predictions',
                                                                                       's-19700101-000008'))

        ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(os.path.join(os.environ['DATASET_BUCKET'],
                                                                            '{}/partitions/data'.format(
                                                                                os.environ['DATASET_ID'])))
        ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(os.path.join(os.environ['DATASET_BUCKET'],
                                                                                       os.environ['DATASET_ID'],
                                                                                       'profile'))

        dask_validate(dask_client=dask_client, vision_definition=test_vd_1['vision_definition'],
                      vision_id=os.environ['VISION_ID'],
                      divina_directory=os.environ['VISION_BUCKET'])

        with open(os.path.join(os.environ['VISION_BUCKET'], 'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']),
                               'metrics.json'), 'r') as f:
            metrics = json.load(f)

        assert (metrics == test_metrics_1)
    finally:
        shutil.rmtree(os.environ['VISION_BUCKET'], ignore_errors=True)
        shutil.rmtree(os.environ['DATASET_BUCKET'], ignore_errors=True)
