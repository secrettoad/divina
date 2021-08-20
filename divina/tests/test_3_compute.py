import pytest
from unittest.mock import patch
import os
from ..cli.cli import cli_build_dataset, cli_train_vision, cli_predict_vision, cli_validate_vision
import pygit2
import pathlib
import dask.dataframe as ddf
import pandas as pd
import joblib
from divina.divina.models.utils import compare_sk_models
import json


@pytest.fixture(scope='session')
def git_sha():
    repo = pygit2.Repository('.')
    repo.index.add_all()
    repo.index.write()
    reference = 'refs/HEAD/{}'.format(repo.head.name)
    message = 'WIP:Testing'
    tree = repo.index.write_tree()
    author = pygit2.Signature(os.environ['GIT_USER'], os.environ['GIT_EMAIL'])
    commiter = pygit2.Signature(os.environ['GIT_USER'], os.environ['GIT_EMAIL'])
    oid = repo.create_commit(reference, author, commiter, message, tree, [repo.head.target])
    repo.head.set_target(oid)
    remote = repo.remotes['origin']
    keypair = pygit2.Keypair('git', pathlib.Path(os.environ['GIT_KEY_PUB']).absolute(),
                             pathlib.Path(os.environ['GIT_KEY']).absolute(), os.environ['GIT_KEY_PASS'])
    callbacks = pygit2.RemoteCallbacks(credentials=keypair)
    remote.push([repo.head.name], callbacks=callbacks)
    return oid.hex


@patch.dict(os.environ, {"DATASET_BUCKET": "s3://divina-test/dataset"})
@patch.dict(os.environ, {"DATASET_ID": 'test1'})
@patch.dict(os.environ, {"DATA_BUCKET": "s3://divina-test/data"})
def test_build_dataset_small(s3_fs, test_df_1, git_sha):
    cli_build_dataset(commit=git_sha, dataset_name=os.environ['DATASET_ID'], write_path=os.environ["DATASET_BUCKET"],
                      read_path=os.environ["DATA_BUCKET"], ec2_keypair_name='divina_ec2_key')
    pd.testing.assert_frame_equal(ddf.read_parquet(
        os.path.join(os.environ['DATASET_BUCKET'], os.environ['DATASET_ID'], 'data')).compute(),
                                  ddf.read_parquet(
                                      os.path.join('stubs', os.environ['DATASET_ID'], 'data')).compute())
    pd.testing.assert_frame_equal(ddf.read_parquet(
        os.path.join(os.environ['DATASET_BUCKET'], os.environ['DATASET_ID'], 'profile')).compute(),
                                  ddf.read_parquet(
                                      os.path.join('stubs', os.environ['DATASET_ID'], 'profile')).compute())


@patch.dict(os.environ, {"VISION_ID": 'test1'})
@patch.dict(os.environ, {"VISION_BUCKET": "s3://divina-test/vision"})
def test_train_small(s3_fs, test_df_1, test_vd_3):
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(test_vd_3['vision_definition']['dataset_directory'],
                     test_vd_3['vision_definition']['dataset_id'], 'data'))
    ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(
        os.path.join(test_vd_3['vision_definition']['dataset_directory'],
                     test_vd_3['vision_definition'][
                         'dataset_id'],
                     'profile'))
    vision_definition = test_vd_3['vision_definition']
    cli_train_vision(vision_definition=vision_definition, write_path=os.environ['VISION_BUCKET'],
                     vision_name=os.environ['VISION_ID'], keep_instances_alive=False, local=False, dask_address=None,
                     ec2_keypair_name='divina_ec2_key')
    assert (compare_sk_models(joblib.load(
        os.path.join(os.environ['VISION_BUCKET'], os.environ['VISION_ID'],
                     'models', 's-19700101-000008_h-1')), joblib.load(
        os.path.join('stubs', os.environ['VISION_ID'],
                     'models', 's-19700101-000008_h-1'))))


@patch.dict(os.environ, {"VISION_ID": 'test1'})
@patch.dict(os.environ, {"VISION_BUCKET": "s3://divina-test/vision"})
def test_predict_small(s3_fs, test_df_1, test_vd_3):
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(test_vd_3['vision_definition']['dataset_directory'],
                     test_vd_3['vision_definition']['dataset_id'], 'data'))
    ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(
        os.path.join(test_vd_3['vision_definition']['dataset_directory'],
                     test_vd_3['vision_definition'][
                         'dataset_id'],
                     'profile'))
    s3_fs.put(os.path.join('stubs', os.environ['VISION_ID'],
                           'models', 's-19700101-000008_h-1'), os.path.join(os.environ['VISION_BUCKET'],
                                                                            os.environ['VISION_ID'],
                                                                            'models', 's-19700101-000008_h-1'),
              recursive=True)
    vision_definition = test_vd_3['vision_definition']
    cli_predict_vision(s3_fs=s3_fs, vision_definition=vision_definition, write_path=os.environ['VISION_BUCKET'],
                       vision_name=os.environ['VISION_ID'], keep_instances_alive=False, local=False, dask_address=None,
                       ec2_keypair_name='divina_ec2_key')
    pd.testing.assert_frame_equal(ddf.read_parquet(
        os.path.join(os.environ['VISION_BUCKET'], os.environ['VISION_ID'],
                     'predictions', 's-19700101-000008')).compute(), ddf.read_parquet(
        os.path.join('stubs', os.environ['VISION_ID'],
                     'predictions', 's-19700101-000008')).compute())


@patch.dict(os.environ, {"VISION_ID": "test1"})
@patch.dict(os.environ, {"VISION_BUCKET": "s3://divina-test/vision"})
def test_validate_small(s3_fs, test_vd_3, test_df_1, test_metrics_1, account_number, dask_client):
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(test_vd_3['vision_definition']['dataset_directory'],
                     test_vd_3['vision_definition']['dataset_id'], 'data'))
    ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(
        os.path.join(test_vd_3['vision_definition']['dataset_directory'],
                     test_vd_3['vision_definition']['dataset_id'],
                     'profile'))

    s3_fs.put(os.path.join('stubs', os.environ['VISION_ID'],
                           'predictions', 's-19700101-000008'), os.path.join(os.environ['VISION_BUCKET'],

                                                                             os.environ['VISION_ID'],
                                                                             'predictions',
                                                                             's-19700101-000008'), recursive=True)

    cli_validate_vision(s3_fs, vision_definition=test_vd_3['vision_definition'], write_path=os.environ['VISION_BUCKET'],
                        vision_name=os.environ['VISION_ID'], ec2_keypair_name='divina_ec2_key',
                        keep_instances_alive=False,
                        local=False, debug=True, dask_address=None)

    with s3_fs.open(os.path.join(os.environ['VISION_BUCKET'], os.environ['VISION_ID'],
                                 'metrics.json'), 'r') as f:
        metrics = json.load(f)

    assert (metrics == test_metrics_1)
