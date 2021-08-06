from ..vision import create_vision
from ...ops import build_publish_dev
import pytest
from unittest.mock import patch
import os
from ..dataset import create_partitioning_ec2
import paramiko
from ...aws.aws_backoff import stop_instances
from ..dataset import build_dataset
from dask import dataframe as ddf
import pandas as pd
import pathlib
import json
from ...models.utils import compare_sk_models
from dask_ml.linear_model import LinearRegression
from ..train import dask_train
import joblib


@pytest.mark.skip('WIP')
def test_e2e_small(divina_test_version, vision_codeartifact):
    build_publish_dev.main()
    codeartifact_token = vision_codeartifact.get_authorization_token(domain='coysu')['authorizationToken']
    divina_pip_arguments = '-i https://aws:{}@coysu-169491045780.d.codeartifact.us-west-2.amazonaws.com/pypi/divina/simple/ --extra-index-url https://www.pypi.org/simple'.format(
        codeartifact_token)
    create_vision(keep_instances_alive=True, import_bucket='coysu-divina-prototype-small',
                  divina_version=divina_test_version, ec2_keyfile='divina-dev', verbosity=3,
                  divina_pip_arguments=divina_pip_arguments)


@patch.dict(os.environ, {"VISION_ID": "test1"})
@patch.dict(os.environ, {"DATA_BUCKET": "s3://divina-test/data"})
def test_dataset_infrastructure(s3_fs, test_df_1, divina_session, divina_test_version):
    test_df_1.to_csv(
        os.path.join(os.environ['DATA_BUCKET'], 'test_df_1.csv'), index=False)
    pricing = divina_session.client('pricing', 'us-east-1')
    ec2 = divina_session.client('ec2', 'us-east-2')
    instance, paramiko_key = create_partitioning_ec2(s3_fs=s3_fs, ec2_client=ec2,
                                                     pricing_client=pricing,
                                                     data_directory=os.environ['DATA_BUCKET'])
    try:
        assert (all([x in instance for x in ['ImageId', 'InstanceId']]) and instance['State'][
            'Name'] == 'running' and type(
            paramiko_key) == paramiko.rsakey.RSAKey)
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


@patch.dict(os.environ, {"DATA_BUCKET": "s3://divina-test/data"})
@patch.dict(os.environ, {"DATASET_BUCKET": "s3://divina-test/dataset"})
@patch.dict(os.environ, {"DATASET_ID": "test1"})
def test_train(test_df_1, test_vd_1, dask_client, account_number):
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(os.path.join(os.environ['DATASET_BUCKET'],
                                                                        os.environ['DATASET_ID'], 'data'))
    ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(os.path.join(os.environ['DATASET_BUCKET'],
                                                                                   os.environ['DATASET_ID'],
                                                                                   'profile'))
    with open(os.path.join(os.environ['DATASET_BUCKET'], '{}'.format(os.environ['DATASET_ID']),
                           'vision_definition.json'), 'w+') as f:
        json.dump(test_vd_1, f)
    models = dask_train(dask_client=dask_client, dask_model=LinearRegression(),
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

    assert (compare_sk_models(joblib.load(os.path.abspath(
        os.path.join(os.environ['VISION_BUCKET'], 'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']),
                     'models', 's-19700101-000008_h-1'))), joblib.load(
        os.path.abspath(os.path.join('stubs', 'coysu-divina-prototype-{}'.format(os.environ['VISION_ID']),
                                     'models', 's-19700101-000008_h-1')))))


def test_predict():
    pass


def test_validate():
    pass
