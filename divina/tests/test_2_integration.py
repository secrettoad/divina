from unittest.mock import patch
import os
from ..dataset import create_partitioning_ec2
from ..aws.aws_backoff import stop_instances
from ..dataset import build_dataset
from dask import dataframe as ddf
import pandas as pd
import pathlib
import json
from divina.divina.models.utils import compare_sk_models
from dask_ml.linear_model import LinearRegression
from ..train import dask_train
import joblib
from ..predict import dask_predict
from ..validate import dask_validate


@patch.dict(os.environ, {"VISION_ID": "test1"})
@patch.dict(os.environ, {"DATA_BUCKET": "s3://divina-test/data"})
def test_dataset_infrastructure(s3_fs, test_df_1, divina_session, divina_test_version):
    test_df_1.to_csv(
        os.path.join(os.environ['DATA_BUCKET'], 'test_df_1.csv'), index=False)
    ec2 = divina_session.client('ec2', 'us-east-2')
    pricing_client = divina_session.client('pricing', region_name='us-east-1')
    instance = create_partitioning_ec2(s3_fs=s3_fs, ec2_client=ec2, pricing_client=pricing_client,
                                                     data_directory=os.environ['DATA_BUCKET'])
    try:
        assert (all([x in instance for x in ['ImageId', 'InstanceId']]) and instance['State'][
            'Name'] == 'running')
    finally:
        stop_instances(instance_ids=[instance['InstanceId']], ec2_client=ec2)


@patch.dict(os.environ, {"DATA_BUCKET": "s3://divina-test/data"})
@patch.dict(os.environ, {"DATASET_BUCKET": "s3://divina-test/dataset"})
@patch.dict(os.environ, {"DATASET_ID": "test1"})
def test_dataset_build(s3_fs, test_df_1):
    test_df_1.to_csv(
        os.path.join(os.environ['DATA_BUCKET'], 'test_df_1.csv'), index=False)
    build_dataset(s3_fs=s3_fs, dataset_directory=os.environ['DATASET_BUCKET'], data_directory=os.environ['DATA_BUCKET'],
                  dataset_id=os.environ['DATASET_ID'])

    pd.testing.assert_frame_equal(ddf.read_parquet(
        os.path.join(os.environ['DATASET_BUCKET'], os.environ['DATASET_ID'], 'data')).compute(),
                                  ddf.read_parquet(
                                      os.path.join('stubs', os.environ['DATASET_ID'], 'data')).compute())
    pd.testing.assert_frame_equal(ddf.read_parquet(
        os.path.join(os.environ['DATASET_BUCKET'], os.environ['DATASET_ID'], 'profile')).compute(),
                                  ddf.read_parquet(
                                      os.path.join('stubs', os.environ['DATASET_ID'], 'profile')).compute())


@patch.dict(os.environ, {"VISION_BUCKET": "s3://divina-test/vision"})
@patch.dict(os.environ, {"VISION_ID": "test1"})
def test_train(test_df_1, test_vd_1, dask_client, account_number):
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(test_vd_1['vision_definition']['dataset_directory'],
                     test_vd_1['vision_definition']['dataset_id'], 'data'))
    ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(
        os.path.join(test_vd_1['vision_definition']['dataset_directory'],
                     test_vd_1['vision_definition'][
                         'dataset_id'],
                     'profile'))
    with open(os.path.join(test_vd_1['vision_definition']['dataset_directory'], test_vd_1['vision_definition']['dataset_id'],
                           'vision_definition.json'), 'w+') as f:
        json.dump(test_vd_1, f)
    dask_train(dask_client=dask_client, dask_model=LinearRegression(),
               vision_definition=test_vd_1['vision_definition'],
               vision_id=os.environ['VISION_ID'], divina_directory=os.environ['VISION_BUCKET'])
    pathlib.Path(
        pathlib.Path(__file__).resolve().parent, 'stubs',
        'coysu-divina-prototype-{}'.format(os.environ['VISION_ID'])).mkdir(
        parents=True, exist_ok=True)

    assert (compare_sk_models(joblib.load(os.path.abspath(
        os.path.join(os.environ['VISION_BUCKET'], os.environ['VISION_ID'],
                     'models', 's-19700101-000008_h-1'))), joblib.load(
        os.path.abspath(os.path.join('stubs', os.environ['VISION_ID'],
                                     'models', 's-19700101-000008_h-1')))))


@patch.dict(os.environ, {"VISION_ID": "test1"})
@patch.dict(os.environ, {"VISION_BUCKET": "s3://divina-test/vision"})
def test_predict(s3_fs, dask_client, test_df_1, test_vd_1, test_model_1, account_number):
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(test_vd_1['vision_definition']['dataset_directory'],
                     test_vd_1['vision_definition']['dataset_id'], 'data'))
    ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(
        os.path.join(test_vd_1['vision_definition']['dataset_directory'],
                     test_vd_1['vision_definition']['dataset_id'],
                     'profile'))
    s3_fs.put(os.path.join('stubs', os.environ['VISION_ID'],
                              'models', 's-19700101-000008_h-1'), os.path.join(os.environ['VISION_BUCKET'],
                                                                               os.environ['VISION_ID'],
                                                                               'models', 's-19700101-000008_h-1'))

    dask_predict(s3_fs=s3_fs, dask_client=dask_client, vision_definition=test_vd_1['vision_definition'],
                 vision_id=os.environ['VISION_ID'],
                 divina_directory=os.environ['VISION_BUCKET'])

    pd.testing.assert_frame_equal(ddf.read_parquet(
        os.path.join(os.environ['VISION_BUCKET'], os.environ['VISION_ID'],
                     'predictions', 's-19700101-000008')).compute(), ddf.read_parquet(
        os.path.join('stubs', os.environ['VISION_ID'],
                     'predictions', 's-19700101-000008')).compute())


@patch.dict(os.environ, {"VISION_ID": "test1"})
@patch.dict(os.environ, {"VISION_BUCKET": "s3://divina-test/vision"})
def test_validate(s3_fs, test_vd_1, test_df_1, test_metrics_1, account_number, dask_client):
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(test_vd_1['vision_definition']['dataset_directory'],
                     test_vd_1['vision_definition']['dataset_id'], 'data'))
    ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(
        os.path.join(test_vd_1['vision_definition']['dataset_directory'],
                     test_vd_1['vision_definition']['dataset_id'],
                     'profile'))

    s3_fs.put(os.path.join('stubs', os.environ['VISION_ID'],
                                 'predictions', 's-19700101-000008'), os.path.join(os.environ['VISION_BUCKET'],

                                                                                   os.environ['VISION_ID'],
                                                                                   'predictions',
                                                                                   's-19700101-000008'))

    dask_validate(s3_fs=s3_fs, dask_client=dask_client, vision_definition=test_vd_1['vision_definition'],
                  vision_id=os.environ['VISION_ID'],
                  divina_directory=os.environ['VISION_BUCKET'])

    with s3_fs.open(os.path.join(os.environ['VISION_BUCKET'], os.environ['VISION_ID'],
                           'metrics.json'), 'r') as f:
        metrics = json.load(f)

    assert (metrics == test_metrics_1)
