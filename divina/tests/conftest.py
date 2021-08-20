import pytest
import pandas as pd
import numpy as np
import boto3
import moto
import pkg_resources
from .stubs import pricing_stubs
from dask.distributed import Client
import pathlib
import os
import s3fs
from unittest.mock import patch

@pytest.fixture(autouse=True)
def run_before_and_after_tests(s3_fs):
    try:
        s3_fs.mkdir('s3://divina-test', region_name=os.environ['AWS_DEFAULT_REGION'])
    except FileExistsError:
        s3_fs.rm('s3://divina-test', recursive=True)
        s3_fs.mkdir('s3://divina-test', region_name=os.environ['AWS_DEFAULT_REGION'])
    '''try:
        os.mkdir('divina-test')
    except FileExistsError:
        shutil.rmtree('divina-test')
        os.mkdir('divina-test')'''
    yield
    try:
        s3_fs.rm('s3://divina-test', recursive=True)
    except FileNotFoundError:
        pass
    '''try:
        shutil.rmtree('divina-test')
    except FileNotFoundError:
        pass'''


@patch.dict(os.environ, {'AWS_SHARED_CREDENTIALS_FILE': '~/.aws/credentials'})
@pytest.fixture()
def divina_session():
    return boto3.Session()


@pytest.fixture(autouse=True)
def reset_s3():
    pass


@pytest.fixture(autouse=True)
def reset_local_filesystem():
    pass


@pytest.fixture()
def s3_fs():
    return s3fs.S3FileSystem()


@pytest.fixture()
def dask_client(request):
    client = Client()
    request.addfinalizer(lambda: client.close())
    return client


@pytest.fixture()
def dask_client(request):
    client = Client()
    request.addfinalizer(lambda: client.close())
    return client


@pytest.fixture()
def test_model_1():
    return os.path.join(pathlib.Path(__file__).resolve().parent, 'stubs', 'models', 'test1')


@pytest.fixture()
def vd_no_dataset_id():
    return {"vision_definition": {
        "target": "c",
        "time_index": 'test1',
        "dataset_id": 'test1'
    }
    }


@pytest.fixture()
def vd_no_dataset_directory():
    return {"vision_definition": {
        "target": "c",
        "time_index": 'test1',
        "dataset_directory": 'divina-test/dataset'
    }
    }


@pytest.fixture()
def vd_no_time_index():
    return {"vision_definition": {
        "target": "c",
        "dataset_directory": 'divina-test/dataset',
        "dataset_id": 'test1'
    }
    }


@pytest.fixture()
def vd_no_target():
    return {"vision_definition": {
        "time_index": "a",
        "dataset_directory": 'divina-test/dataset',
        "dataset_id": 'test1'
    }
    }


@pytest.fixture()
def vd_time_validation_splits_not_list():
    return {"vision_definition": {
        "time_index": "a",
        "target": "c",
        "time_validation_splits": "1970-01-01 00:00:08",
        "time_horizons": [1],
        "dataset_directory": 'divina-test/dataset',
        "dataset_id": 'test1'
    }
    }


@pytest.fixture()
def vd_time_horizons_not_list():
    return {"vision_definition": {
        "time_index": "a",
        "target": "c",
        "time_validation_splits": ["1970-01-01 00:00:08"],
        "time_horizons": 1,
        "dataset_directory": 'divina-test/dataset',
        "dataset_id": 'test1'
    }
    }


@pytest.fixture()
def test_metrics_1():
    return {'splits': {'1970-01-01 00:00:08': {'time_horizons': {'1': {'mae': 8.40376381901896}}}}}


@pytest.fixture()
def test_vd_1():
    return {"vision_definition": {
        "time_index": "a",
        "target": "c",
        "time_validation_splits": ["1970-01-01 00:00:08"],
        "time_horizons": [1],
        "dataset_directory": 'divina-test/dataset',
        "dataset_id": 'test1'
    }
    }


@pytest.fixture()
def test_vd_2():
    return {"vision_definition": {
        "time_index": "a",
        "target": "c",
        "time_validation_splits": ["1970-01-01 00:00:08"],
        "time_horizons": [1],
        "dataset_directory": 'divina-test/dataset',
        "dataset_id": "test1",
        "joins": [{'dataset_directory': 'dataset', 'dataset_id': 'test2', 'join_on': ('a', 'test2_a')}]
    }
    }


@pytest.fixture()
def test_vd_3():
    return {"vision_definition": {
        "time_index": "a",
        "target": "c",
        "time_validation_splits": ["1970-01-01 00:00:08"],
        "time_horizons": [1],
        "dataset_directory": 's3://divina-test/dataset',
        "dataset_id": 'test1'
    }
    }


@pytest.fixture()
def test_composite_dataset_1():
    df = pd.DataFrame([[4.0, 5.0, 6.0, 4.0, np.NaN, 6.0], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0], [6.0, 5.0, 6.0, np.NaN, np.NaN, np.NaN], [4.0, 5.0, 6.0, 4.0, np.NaN, 6.0], [10.0, 11.0, 12.0, np.NaN, np.NaN, np.NaN], [7.0, 8.0, 9.0, 7.0, 8.0, np.NaN], [4.0, 5.0, 6.0, 4.0, np.NaN, 6.0], [5.0, 5.0, 6.0, np.NaN, np.NaN, np.NaN], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, np.NaN, np.NaN, np.NaN], [7.0, 8.0, 9.0, 7.0, 8.0, np.NaN], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, np.NaN, np.NaN, np.NaN], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, np.NaN, np.NaN, np.NaN], [7.0, 8.0, 9.0, 7.0, 8.0, np.NaN], [10.0, 11.0, 12.0, np.NaN, np.NaN, np.NaN], [7.0, 8.0, 9.0, 7.0, 8.0, np.NaN], [5.0, 5.0, 6.0, np.NaN, np.NaN, np.NaN], [7.0, 8.0, 9.0, 7.0, 8.0, np.NaN], [4.0, 5.0, 6.0, 4.0, np.NaN, 6.0], [5.0, 5.0, 6.0, np.NaN, np.NaN, np.NaN], [10.0, 11.0, 12.0, np.NaN, np.NaN, np.NaN], [10.0, 11.0, 12.0, np.NaN, np.NaN, np.NaN], [7.0, 8.0, 9.0, 7.0, 8.0, np.NaN]]
                      )
    df.columns = ['a', 'b', 'c', 'test2_a', 'test2_e', 'test2_f']
    return df


@pytest.fixture()
def test_composite_profile_1():
    df = pd.DataFrame([['25%', 4.0, 5.0, 6.0, 2.5, 5.0, 4.5],
                       ['50%', 7.0, 8.0, 9.0, 4.0, 8.0, 6.0],
                       ['75%', 10.0, 11.0, 12.0, 5.5, 9.5, 9.0],
                       ['count', 25.0, 25.0, 25.0, 3.0, 3.0, 3.0],
                       ['max', 10.0, 11.0, 12.0, 7.0, 11.0, 12.0],
                       ['mean', 6.12, 6.92, 7.92, 4.0, 7.0, 7.0],
                       ['min', 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                       ['std', 3.1400636936215163, 3.2264531609803355, 3.2264531609803355, 3.0, 4.58257569495584,
                        4.58257569495584]])
    df.columns = ['statistic', 'a', 'b', 'c', 'test2_a', 'test2_e', 'test2_f']
    df.set_index('statistic', inplace=True)
    df.index.name = None
    return df


@pytest.fixture()
def test_df_1():
    df = pd.DataFrame([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [5.0, 5.0, 6.0],
                       [6.0, 5.0, 6.0],
                       [7.0, 8.0, 9],
                       [10.0, 11.0, 12.0]]).sample(25, replace=True, random_state=11).reset_index(drop=True)
    df.columns = ['a', 'b', 'c']
    return df


@pytest.fixture()
def test_df_2():
    df = pd.DataFrame([[1, 2.0, 3.0],
                       [4, np.NaN, 6.0],
                       [7, 8.0, np.NaN],
                       [np.NaN, 11.0, 12.0]])
    df.columns = ['a', 'e', 'f']
    return df


@pytest.fixture()
def test_df_3():
    df = pd.DataFrame([[1, 2, 3],
                       [4, 'a', 6],
                       [7, 8, 'b'],
                       ['c', 11, 12]]).astype('str')
    df.columns = ['a', 'b', 'c']
    return df


@pytest.fixture()
def source_sts():
    with moto.mock_sts():
        yield boto3.client("sts", region_name='us-east-2')


@pytest.fixture()
def source_s3():
    with moto.mock_s3():
        yield boto3.client("s3", region_name='us-east-2')


@pytest.fixture()
def source_iam():
    with moto.mock_iam():
        yield boto3.client("iam", region_name='us-east-2')


@pytest.fixture()
def vision_sts():
    with moto.mock_sts():
        yield boto3.client("sts", region_name='us-east-2')


@pytest.fixture()
def vision_s3():
    with moto.mock_s3():
        yield boto3.client("s3", region_name='us-east-2')


@pytest.fixture()
def vision_ec2():
    with moto.mock_ec2():
        yield boto3.client("ec2", region_name='us-east-2')


@pytest.fixture()
def vision_emr():
    with moto.mock_emr():
        yield boto3.client("emr", region_name='us-east-2')


@pytest.fixture()
def vision_iam():
    with moto.mock_iam():
        yield boto3.client("iam", region_name='us-east-2')


@pytest.fixture()
def divina_test_version():
    return pkg_resources.get_distribution('divina').version


@pytest.fixture()
def vision_codeartifact():
    return boto3.client('codeartifact', region_name='us-west-2')


@pytest.fixture()
def account_number(monkeypatch):
    monkeypatch.setenv('ACCOUNT_NUMBER', '123456789012')


@pytest.fixture()
def ec2_pricing_stub():
    return pricing_stubs.ec2_pricing_stub
