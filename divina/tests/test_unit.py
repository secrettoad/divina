import os
import json
from unittest.mock import patch, MagicMock
from ..vision import create_partitioning_ec2, create_modelling_emr, validate_vision_definition
from ..train import dask_train
from ..predict import dask_predict
from ..dataset import build_dataset, get_dataset
from ..validate import dask_validate
from ..errors import InvalidDataDefinitionException
import boto3
import paramiko
import shutil
import pathlib
from dask_ml.linear_model import LinearRegression
from divina.divina.models.utils import compare_sk_models
import joblib
import pandas as pd
import dask.dataframe as ddf


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
@patch('s3fs.S3FileSystem.ls', lambda s, x: [os.path.join(x, y) for y in os.listdir(x)])
@patch('s3fs.S3FileSystem.info', lambda s, x: {'size': os.path.getsize(x)})
@patch.dict(os.environ, {"VISION_ID": "test1"})
@patch.dict(os.environ, {"DATA_BUCKET": "divina-test/data"})
def test_dataset_infrastructure(s3_fs, test_df_1, vision_ec2, divina_test_version, ec2_pricing_stub,
                                account_number):
    pathlib.Path(
        os.environ['DATA_BUCKET']).mkdir(
        parents=True, exist_ok=True)
    test_df_1.to_csv(
        os.path.join(os.environ['DATA_BUCKET'], 'test_df_1.csv'), index=False)
    with patch('boto3.client'):
        mock_vision_pricing = boto3.client('pricing')
    mock_vision_pricing.get_products.return_value = ec2_pricing_stub
    instance, paramiko_key = create_partitioning_ec2(s3_fs, ec2_client=vision_ec2,
                                                     pricing_client=mock_vision_pricing,
                                                     data_directory=os.environ['DATA_BUCKET'])
    assert (all([x in instance for x in ['ImageId', 'InstanceId']]) and instance['State']['Name'] == 'running' and type(
        paramiko_key) == paramiko.rsakey.RSAKey)


@patch('s3fs.S3FileSystem.open', open)
@patch('s3fs.S3FileSystem.ls', os.listdir)
@patch.dict(os.environ, {"DATA_BUCKET": "divina-test/data-bucket"})
@patch.dict(os.environ, {"DATASET_BUCKET": "divina-test/dataset-bucket"})
@patch.dict(os.environ, {"DATASET_ID": "test1"})
def test_dataset_build(s3_fs, vision_s3, test_df_1, account_number):
    try:
        pathlib.Path(
            os.environ['DATA_BUCKET']).mkdir(
            parents=True, exist_ok=True)
        test_df_1.to_csv(
            os.path.join(os.environ['DATA_BUCKET'], 'test_df_1.csv'), index=False)
        pathlib.Path(
            os.environ['DATASET_BUCKET'], os.environ['DATASET_ID']).mkdir(
            parents=True, exist_ok=True)
        build_dataset(s3_fs=s3_fs, dataset_directory=os.environ['DATASET_BUCKET'], data_directory=os.environ['DATA_BUCKET'],
                      dataset_id=os.environ['DATASET_ID'])

        '''Uncomment below to generate stub datasets'''
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
@patch.dict(os.environ, {"VISION_BUCKET": "divina-test/vision"})
@patch.dict(os.environ, {"VISION_ID": "test1"})
def test_dask_train(test_df_1, test_vd_1, dask_client, account_number):
    try:
        pathlib.Path(
            os.path.join(os.environ['VISION_BUCKET'], '{}'.format(os.environ['VISION_ID']),
                         'models')).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(
            os.path.join(test_vd_1['vision_definition']['dataset_directory'], '{}'.format(test_vd_1['vision_definition']['dataset_id']),
                         'data')).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(
            os.path.join(test_vd_1['vision_definition']['dataset_directory'], '{}'.format(test_vd_1['vision_definition']['dataset_id']),
                         'profile')).mkdir(
            parents=True, exist_ok=True)
        ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(os.path.join(test_vd_1['vision_definition']['dataset_directory'],
                                                                            test_vd_1['vision_definition']['dataset_id'], 'data'))
        ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(os.path.join(test_vd_1['vision_definition']['dataset_directory'],
                                                                                       test_vd_1['vision_definition'][
                                                                                           'dataset_id'],
                                                                                       'profile'))
        models = dask_train(dask_client=dask_client, dask_model=LinearRegression(),
                            vision_definition=test_vd_1['vision_definition'],
                            vision_id=os.environ['VISION_ID'], divina_directory=os.environ['VISION_BUCKET'])
        pathlib.Path(
            pathlib.Path(__file__).resolve().parent, 'stubs',
            os.environ['VISION_ID']).mkdir(
            parents=True, exist_ok=True)

        '''Uncomment below to generate stub models'''

        pathlib.Path(pathlib.Path(pathlib.Path(__file__).resolve().parent, 'stubs',
                                  os.environ['VISION_ID'], 'models')).mkdir(
            parents=True, exist_ok=True)
        for model in models:
            joblib.dump(models[model],
                        str(pathlib.Path(pathlib.Path(__file__).resolve().parent, 'stubs',
                                         os.environ['VISION_ID'], 'models', model)))

        '''end stubbing'''

        assert (compare_sk_models(joblib.load(os.path.abspath(
            os.path.join(os.environ['VISION_BUCKET'], os.environ['VISION_ID'],
                         'models', 's-19700101-000008_h-1'))), joblib.load(
            os.path.abspath(os.path.join('stubs', os.environ['VISION_ID'],
                                         'models', 's-19700101-000008_h-1')))))
    finally:
        shutil.rmtree(os.environ['VISION_BUCKET'], ignore_errors=True)
        shutil.rmtree(test_vd_1['vision_definition']['dataset_directory'], ignore_errors=True)


@patch('s3fs.S3FileSystem.open', open)
@patch('s3fs.S3FileSystem.ls', os.listdir)
def test_get_composite_dataset(test_df_1, test_df_2, test_vd_2,
                               test_composite_profile_1, test_composite_df_1, dask_client):
    try:
        for path in ['data', 'profile']:
            for dataset in test_vd_2['vision_definition']['joins']:
                pathlib.Path(
                    os.path.join(dataset['dataset_directory'], '{}'.format(dataset['dataset_id']),
                                 path)).mkdir(
                    parents=True, exist_ok=True)
            pathlib.Path(
                os.path.join(test_vd_2['vision_definition']['dataset_directory'], '{}'.format(test_vd_2['vision_definition']['dataset_id']),
                             path)).mkdir(
                parents=True, exist_ok=True)
        ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(os.path.join(test_vd_2['vision_definition']['dataset_directory'],
                                                                            test_vd_2['vision_definition']['dataset_id'], 'data'))
        ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(os.path.join(test_vd_2['vision_definition']['dataset_directory'],
                                                                                       test_vd_2['vision_definition']['dataset_id'],
                                                                                       'profile'))
        ddf.from_pandas(test_df_2, chunksize=10000).to_parquet(os.path.join(test_vd_2['vision_definition']['joins'][0]['dataset_directory'],
                                                                            test_vd_2['vision_definition']['joins'][0]['dataset_id'], 'data'))
        ddf.from_pandas(test_df_2.describe(), chunksize=10000).to_parquet(os.path.join(test_vd_2['vision_definition']['joins'][0]['dataset_directory'],
                                                                                       test_vd_2['vision_definition']['joins'][0]['dataset_id'],
                                                                                       'profile'))
        df, profile = get_dataset(test_vd_2['vision_definition'])

        pd.testing.assert_frame_equal(df.compute(), test_composite_df_1)
        pd.testing.assert_frame_equal(profile.compute(), test_composite_profile_1)
    finally:
        shutil.rmtree(test_vd_2['vision_definition']['dataset_directory'], ignore_errors=True)


@patch('s3fs.S3FileSystem.open', open)
@patch('s3fs.S3FileSystem.ls', os.listdir)
@patch.dict(os.environ, {"VISION_ID": "test1"})
@patch.dict(os.environ, {"VISION_BUCKET": "divina-test/vision"})
def test_dask_predict(s3_fs, dask_client, test_df_1, test_vd_1, test_model_1, account_number):
    try:
        pathlib.Path(
            os.path.join(test_vd_1['vision_definition']['dataset_directory'], test_vd_1['vision_definition']['dataset_id'], 'data')).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(
            os.path.join(os.environ['VISION_BUCKET'], os.environ['VISION_ID'],
                         'models')).mkdir(
            parents=True, exist_ok=True)
        ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(os.path.join(test_vd_1['vision_definition']['dataset_directory'],
                                                                            test_vd_1['vision_definition']['dataset_id'], 'data'))
        ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(os.path.join(test_vd_1['vision_definition']['dataset_directory'],
                                                                                       test_vd_1['vision_definition']['dataset_id'],
                                                                                       'profile'))
        shutil.copy2(os.path.join('stubs', os.environ['VISION_ID'],
                                  'models', 's-19700101-000008_h-1'), os.path.join(os.environ['VISION_BUCKET'],
                                                                                       os.environ['VISION_ID'],
                                                                                   'models', 's-19700101-000008_h-1'))
        with open(os.path.join(os.environ['VISION_BUCKET'], os.environ['VISION_ID'],
                               'vision_definition.json'), 'w+') as f:
            json.dump(test_vd_1, f)

        dask_predict(s3_fs=s3_fs, dask_client=dask_client, vision_definition=test_vd_1['vision_definition'],
                     vision_id=os.environ['VISION_ID'],
                     divina_directory=os.environ['VISION_BUCKET'])

        '''Uncomment below to generate stub models'''

        pathlib.Path(pathlib.Path(pathlib.Path(__file__).resolve().parent, 'stubs',
                                  os.environ['VISION_ID'], 'predictions')).mkdir(
            parents=True, exist_ok=True)
        shutil.copytree(
            os.path.join(os.environ['VISION_BUCKET'], os.environ['VISION_ID'],
                         'predictions', 's-19700101-000008'), os.path.join('stubs',

                                                                               os.environ['VISION_ID'],
                                                                           'predictions', 's-19700101-000008'),
            dirs_exist_ok=True)

        '''end stubbing'''

        pd.testing.assert_frame_equal(ddf.read_parquet(
            os.path.join(os.environ['VISION_BUCKET'], os.environ['VISION_ID'],
                         'predictions', 's-19700101-000008')).compute(), ddf.read_parquet(
            os.path.join('stubs', os.environ['VISION_ID'],
                         'predictions', 's-19700101-000008')).compute())
    finally:
        shutil.rmtree(os.environ['VISION_BUCKET'], ignore_errors=True)
        shutil.rmtree(test_vd_1['vision_definition']['dataset_directory'], ignore_errors=True)


@patch('s3fs.S3FileSystem.open', open)
@patch('s3fs.S3FileSystem.ls', os.listdir)
@patch.dict(os.environ, {"VISION_ID": "test1"})
@patch.dict(os.environ, {"VISION_BUCKET": "divina-test/vision"})
def test_dask_validate(s3_fs, test_vd_1, test_df_1, test_metrics_1, account_number, dask_client):
    try:
        pathlib.Path(
            os.path.join(test_vd_1['vision_definition']['dataset_directory'], test_vd_1['vision_definition']['dataset_id'], 'data')).mkdir(
            parents=True, exist_ok=True)
        ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(os.path.join(test_vd_1['vision_definition']['dataset_directory'],
                                                                            test_vd_1['vision_definition']['dataset_id'], 'data'))
        ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(os.path.join(test_vd_1['vision_definition']['dataset_directory'],
                                                                                       test_vd_1['vision_definition']['dataset_id'],
                                                                                       'profile'))
        pathlib.Path(
            os.path.join(os.environ['VISION_BUCKET'], os.environ['VISION_ID'],
                         'predictions')).mkdir(
            parents=True, exist_ok=True)

        shutil.copytree(os.path.join('stubs', os.environ['VISION_ID'],
                                     'predictions', 's-19700101-000008'), os.path.join(os.environ['VISION_BUCKET'],

                                                                                           os.environ['VISION_ID'],
                                                                                       'predictions',
                                                                                       's-19700101-000008'))

        ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(os.path.join(test_vd_1['vision_definition']['dataset_directory'],
                                                                            '{}/data'.format(
                                                                                test_vd_1['vision_definition']['dataset_id'])))
        ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(os.path.join(test_vd_1['vision_definition']['dataset_directory'],
                                                                                       test_vd_1['vision_definition']['dataset_id'],
                                                                                       'profile'))

        dask_validate(s3_fs=s3_fs, dask_client=dask_client, vision_definition=test_vd_1['vision_definition'],
                      vision_id=os.environ['VISION_ID'],
                      divina_directory=os.environ['VISION_BUCKET'])

        with open(os.path.join(os.environ['VISION_BUCKET'], os.environ['VISION_ID'],
                               'metrics.json'), 'r') as f:
            metrics = json.load(f)

        assert (metrics == test_metrics_1)
    finally:
        shutil.rmtree(os.environ['VISION_BUCKET'], ignore_errors=True)
        shutil.rmtree(test_vd_1['vision_definition']['dataset_directory'], ignore_errors=True)
