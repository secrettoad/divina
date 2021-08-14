import pytest
import subprocess
from unittest.mock import patch
import os
from ..cli.cli import cli_build
import pygit2


@pytest.fixture(scope='session')
def git_sha():
    repo = pygit2.Repository('.')
    branch = repo.lookup_branch(os.environ['GIT_BRANCH'])
    ref = repo.lookup_reference(branch.name)
    repo.checkout(ref)
    repo.index.add_all()
    repo.index.write()
    reference = 'refs/HEAD/{}'.format(repo.head.name)
    message = 'WIP:Testing'
    tree = repo.index.write_tree()
    author = pygit2.Signature(os.environ['GIT_USER'], os.environ['GIT_EMAIL'])
    commiter = pygit2.Signature(os.environ['GIT_USER'], os.environ['GIT_EMAIL'])
    hex = repo.create_commit(reference, author, commiter, message, tree, [repo.head.target.hex])
    repo.push('refs/heads/{}'.format(repo.head.name))
    return hex


@patch.dict(os.environ, {"DATASET_BUCKET": "s3://divina-test/dataset"})
@patch.dict(os.environ, {"DATASET_ID": 'test_e2e_1'})
@patch.dict(os.environ, {"DATA_BUCKET": "s3://divina-test/data"})
def test_e2e_small(s3_fs, test_df_1, git_sha):
    cli_build(commit=git_sha, dataset_name=os.environ['DATASET_ID'], write_path=os.environ["DATASET_BUCKET"], read_path=os.environ["DATA_BUCKET"], ec2_keypair_name='divina_ec2_key')
    assert(1 == 1)
