import pytest
import subprocess
from unittest.mock import patch
import os
from ..cli.cli import cli_build


@patch.dict(os.environ, {"DATASET_BUCKET": "s3://divina-test/dataset"})
@patch.dict(os.environ, {"DATASET_ID": 'test_e2e_1'})
@patch.dict(os.environ, {"DATA_BUCKET": "s3://divina-test/data"})
def test_e2e_small(s3_fs, test_df_1):
    cli_build(dataset_name=os.environ['DATASET_ID'], write_path=os.environ["DATASET_BUCKET"], read_path=os.environ["DATA_BUCKET"], ec2_keypair_name='divina_ec2_key')
    assert(1 == 1)
