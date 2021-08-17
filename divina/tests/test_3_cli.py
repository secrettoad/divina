import pytest
import subprocess
from unittest.mock import patch
import os
from ..cli.cli import cli_build_dataset, cli_train_vision
import pygit2
import pathlib
import dask.dataframe as ddf
import pandas as pd

@pytest.mark.skip('WIP')
@pytest.fixture(autouse=True)
def run_before_and_after_tests(tmpdir, s3_fs):
    yield
    for dir in ["s3://divina-test/data", "s3://divina-test/dataset", "s3://divina-test/vision"]:
        s3_fs.rmdir(dir)


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
    ##TODO start here turn these into absolute paths
    keypair = pygit2.Keypair('git', pathlib.Path(os.environ['GIT_KEY_PUB']).absolute(), pathlib.Path(os.environ['GIT_KEY']).absolute(), os.environ['GIT_KEY_PASS'])
    callbacks = pygit2.RemoteCallbacks(credentials=keypair)
    remote.push([repo.head.name], callbacks=callbacks)
    return oid.hex


@patch.dict(os.environ, {"DATASET_BUCKET": "s3://divina-test/dataset"})
@patch.dict(os.environ, {"DATASET_ID": 'test1'})
@patch.dict(os.environ, {"DATA_BUCKET": "s3://divina-test/data"})
def test_build_dataset_small(s3_fs, test_df_1, git_sha):
    cli_build_dataset(commit=git_sha, dataset_name=os.environ['DATASET_ID'], write_path=os.environ["DATASET_BUCKET"], read_path=os.environ["DATA_BUCKET"], ec2_keypair_name='divina_ec2_key')
    pd.testing.assert_frame_equal(ddf.read_parquet(
        os.path.join(os.environ['DATASET_BUCKET'], os.environ['DATASET_ID'], 'data')).compute(),
                                  ddf.read_parquet(
                                      os.path.join('stubs', os.environ['DATASET_ID'], 'data')).compute())
    pd.testing.assert_frame_equal(ddf.read_parquet(
        os.path.join(os.environ['DATASET_BUCKET'], os.environ['DATASET_ID'], 'profile')).compute(),
                                  ddf.read_parquet(
                                      os.path.join('stubs', os.environ['DATASET_ID'], 'profile')).compute())


@patch.dict(os.environ, {"DATASET_BUCKET": "s3://divina-test/dataset"})
@patch.dict(os.environ, {"VISION_ID": 'test1'})
@patch.dict(os.environ, {"VISION_BUCKET": "s3://divina-test/vision"})
def test_train_vision_small(s3_fs, test_df_1, git_sha, test_vd_1):
    cli_train_vision(vision_definition=test_vd_1['vision_definition'], write_path=os.environ['VISION_ID'], vision_name=os.environ['VISION_ID'], keep_instances_alive=False, local=False, dask_address=None, commit=git_sha, ec2_keypair_name='divina_ec2_key')
    assert(1 == 1)
