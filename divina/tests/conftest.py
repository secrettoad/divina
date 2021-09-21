import pytest
import pandas as pd
import numpy as np
import moto
import pkg_resources
from .stubs import pricing_stubs
from dask.distributed import Client
import os
import s3fs
from unittest.mock import patch
import shutil
from dask_ml.linear_model import LinearRegression
import dask.dataframe as ddf
import boto3
from dask_cloudprovider.aws import EC2Cluster
from pandas import Timestamp
import fsspec


@pytest.fixture()
def test_bucket():
    return "s3://divina-test-2"


@pytest.fixture()
def setup_teardown_test_bucket_contents(s3_fs, request, test_bucket):
    fsspec.filesystem('s3').invalidate_cache()
    test_path = "{}/{}".format(test_bucket, request.node.originalname)
    try:
        s3_fs.mkdir(
            test_path,
            region_name=os.environ["AWS_DEFAULT_REGION"],
            acl="private",
        )
    except FileExistsError:
        s3_fs.rm(test_path, recursive=True)
        s3_fs.mkdir(
            test_path,
            region_name=os.environ["AWS_DEFAULT_REGION"],
            acl="private",
        )
    yield
    try:
        s3_fs.rm(test_path, recursive=True)
    except FileNotFoundError:
        pass


@pytest.fixture(autouse=True)
def setup_teardown_test_directory(s3_fs, test_bucket):
    try:
        os.mkdir("divina-test")
    except FileExistsError:
        shutil.rmtree("divina-test")
        os.mkdir("divina-test")
    yield
    try:
        shutil.rmtree("divina-test")
    except FileNotFoundError:
        pass


@patch.dict(os.environ, {"AWS_SHARED_CREDENTIALS_FILE": "~/.aws/credentials"})
@pytest.fixture()
def divina_session():
    return boto3.Session()


@pytest.fixture(autouse=True)
def reset_s3():
    pass


@pytest.fixture(autouse=True)
def reset_local_filesystem():
    pass


@pytest.fixture(scope="session")
def s3_fs():
    return s3fs.S3FileSystem()


@pytest.fixture()
def dask_client(request):
    client = Client()
    request.addfinalizer(lambda: client.close())
    return client


@pytest.fixture(scope="session")
def dask_client_remote():
    cluster = EC2Cluster(
        key_name="divina2",
        security=False,
        docker_image="jhurdle/divina:latest",
        debug=False,
        env_vars={
            "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
            "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
            "AWS_DEFAULT_REGION": os.environ["AWS_DEFAULT_REGION"],
        },
        auto_shutdown=True,
    )
    cluster.adapt(minimum=1, maximum=10)
    client = Client(cluster)
    yield client
    client.shutdown()


@pytest.fixture()
def fd_no_dataset_directory():
    return {
        "forecast_definition": {
            "target": "c",
            "time_index": "a",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": [1],
        }
    }


@pytest.fixture()
def fd_invalid_model():
    return {
        "forecast_definition": {
            "target": "c",
            "time_index": "a",
            "dataset_directory": "divina-test/dataset/test1",
            "model": "scikitlearn.linear_models.linearRegression",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": [1],
        }
    }


@pytest.fixture()
def fd_no_time_index():
    return {
        "forecast_definition": {
            "target": "c",
            "dataset_directory": "divina-test/dataset/test1",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": [1],
        }
    }


@pytest.fixture()
def fd_no_target():
    return {
        "forecast_definition": {
            "time_index": "a",
            "dataset_directory": "divina-test/dataset/test1",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": [1],
        }
    }


@pytest.fixture()
def fd_time_validation_splits_not_list():
    return {
        "forecast_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": "1970-01-01 00:00:08",
            "time_horizons": [1],
            "dataset_directory": "divina-test/dataset/test1",
        }
    }


@pytest.fixture()
def fd_time_horizons_not_list():
    return {
        "forecast_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": 1,
            "dataset_directory": "divina-test/dataset/test1",
        }
    }


@pytest.fixture()
def fd_time_horizons_range_not_tuple():
    return {
        "forecast_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": [[1, 60]],
            "dataset_directory": "divina-test/dataset/test1",
        }
    }


@pytest.fixture()
def test_model_1(test_df_1):
    return LinearRegression().fit(
        ddf.from_pandas(test_df_1, chunksize=10000)[["b"]].to_dask_array(lengths=True),
        ddf.from_pandas(test_df_1, chunksize=10000)["c"],
    )


@pytest.fixture()
def test_params_1(test_model_1):
    return {'params': {'b': test_model_1._coef[0]}}


@pytest.fixture()
def test_params_2(test_model_1):
    return {'params': {'b': test_model_1._coef[0] + 1}}


@pytest.fixture()
def test_metrics_1():
    return {'splits': {'1970-01-01 00:00:07': {'time_horizons': {'1': {'mae': 80.62968620546809}}}}}


@pytest.fixture()
def test_predictions_1():
    df = pd.DataFrame(
        [[Timestamp('1970-01-01 00:00:05'), 15.56372549019609], [Timestamp('1970-01-01 00:00:06'), 5.836397058823538],
         [Timestamp('1970-01-01 00:00:07'), 47.66390931372551], [Timestamp('1970-01-01 00:00:10'), 0.9727328431372619],
         [Timestamp('1970-01-01 00:00:10'), 0.9727328431372619], [Timestamp('1970-01-01 00:00:11'), 0.9727328431372619],
         [Timestamp('1970-01-01 00:00:12'), 0.9727328431372619], [Timestamp('1970-01-01 00:00:13'), 0.9727328431372619],
         [Timestamp('1970-01-01 00:00:14'), 0.9727328431372619], [Timestamp('1970-01-01 00:00:10'), 1.9454656862745172],
         [Timestamp('1970-01-01 00:00:10'), 1.9454656862745172], [Timestamp('1970-01-01 00:00:11'), 1.9454656862745172],
         [Timestamp('1970-01-01 00:00:12'), 1.9454656862745172], [Timestamp('1970-01-01 00:00:13'), 1.9454656862745172],
         [Timestamp('1970-01-01 00:00:14'), 1.9454656862745172], [Timestamp('1970-01-01 00:00:10'), 2.9181985294117725],
         [Timestamp('1970-01-01 00:00:10'), 2.9181985294117725], [Timestamp('1970-01-01 00:00:11'), 2.9181985294117725],
         [Timestamp('1970-01-01 00:00:12'), 2.9181985294117725], [Timestamp('1970-01-01 00:00:13'), 2.9181985294117725],
         [Timestamp('1970-01-01 00:00:14'), 2.9181985294117725], [Timestamp('1970-01-01 00:00:10'), 3.8909313725490273],
         [Timestamp('1970-01-01 00:00:10'), 3.8909313725490273], [Timestamp('1970-01-01 00:00:11'), 3.8909313725490273],
         [Timestamp('1970-01-01 00:00:12'), 3.8909313725490273], [Timestamp('1970-01-01 00:00:13'), 3.8909313725490273],
         [Timestamp('1970-01-01 00:00:14'), 3.8909313725490273], [Timestamp('1970-01-01 00:00:10'), 4.863664215686283],
         [Timestamp('1970-01-01 00:00:10'), 4.863664215686283], [Timestamp('1970-01-01 00:00:11'), 4.863664215686283],
         [Timestamp('1970-01-01 00:00:12'), 4.863664215686283], [Timestamp('1970-01-01 00:00:13'), 4.863664215686283],
         [Timestamp('1970-01-01 00:00:14'), 4.863664215686283], [Timestamp('1970-01-01 00:00:10'), 5.836397058823538],
         [Timestamp('1970-01-01 00:00:10'), 5.836397058823538], [Timestamp('1970-01-01 00:00:11'), 5.836397058823538],
         [Timestamp('1970-01-01 00:00:12'), 5.836397058823538], [Timestamp('1970-01-01 00:00:13'), 5.836397058823538],
         [Timestamp('1970-01-01 00:00:14'), 5.836397058823538]]
    )
    df.columns = ["a", "c_h_1_pred"]
    return df


@pytest.fixture()
def test_fd_1():
    return {
        "forecast_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": ["1970-01-01 00:00:07"],
            "train_val_cutoff": "1970-01-01 00:00:08",
            "forecast_start": "1970-01-01 00:00:05",
            "forecast_end": "1970-01-01 00:00:14",
            "forecast_freq": 'S',
            "scenarios": {'b': {'values': [(0, 5)], 'start': "1970-01-01 00:00:09", 'end': "1970-01-01 00:00:14"}},
            "time_horizons": [1],
            "dataset_directory": "divina-test/dataset/test1",
            "model": "LinearRegression",
        }
    }


@pytest.fixture()
def test_fd_2():
    return {
        "forecast_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": [1],
            "dataset_directory": "divina-test/dataset/test1",
            "joins": [
                {
                    "dataset_directory": "dataset/test2",
                    "join_on": ("a", "a"),
                    "as": "test2",
                }
            ],
        }
    }


@pytest.fixture()
def test_fd_3(test_bucket, test_fd_1):
    test_fd = test_fd_1
    test_fd["forecast_definition"].update({"dataset_directory": "{}/dataset/test1".format(test_bucket)})
    return test_fd


@pytest.fixture()
def test_composite_dataset_1():
    df = pd.DataFrame(
        [[Timestamp('1970-01-01 00:00:01'), 8.0, 12.0, 2.0, 3.0],
         [Timestamp('1970-01-01 00:00:04'), 20.0, 24.0, np.NaN, 6.0],
         [Timestamp('1970-01-01 00:00:05'), 15.0, 18.0, np.NaN, np.NaN],
         [Timestamp('1970-01-01 00:00:06'), 5.0, 6.0, np.NaN, np.NaN],
         [Timestamp('1970-01-01 00:00:07'), 48.0, 54.0, 8.0, np.NaN],
         [Timestamp('1970-01-01 00:00:10'), 77.0, 84.0, np.NaN, np.NaN]]
    )
    df.columns = ["a", "b", "c", "e", "f"]
    return df


@pytest.fixture()
def test_df_1():
    df = (
        pd.DataFrame(
            [
                [Timestamp('1970-01-01 00:00:01'), 2.0, 3.0],
                [Timestamp('1970-01-01 00:00:04'), 5.0, 6.0],
                [Timestamp('1970-01-01 00:00:05'), 5.0, 6.0],
                [Timestamp('1970-01-01 00:00:06'), 5.0, 6.0],
                [Timestamp('1970-01-01 00:00:07'), 8.0, 9],
                [Timestamp('1970-01-01 00:00:10'), 11.0, 12.0],
            ]
        )
            .sample(25, replace=True, random_state=11)
            .reset_index(drop=True)
    )
    df.columns = ["a", "b", "c"]
    return df


@pytest.fixture()
def test_df_2():
    df = pd.DataFrame(
        [[Timestamp('1970-01-01 00:00:01'), 2.0, 3.0], [Timestamp('1970-01-01 00:00:04'), np.NaN, 6.0],
         [Timestamp('1970-01-01 00:00:07'), 8.0, np.NaN], [np.NaN, 11.0, 12.0]]
    )
    df.columns = ["a", "e", "f"]
    return df


@pytest.fixture()
def test_df_3():
    df = pd.DataFrame([[1, 2, 3], [4, "a", 6], [7, 8, "b"], ["c", 11, 12]]).astype(
        "str"
    )
    df.columns = ["a", "b", "c"]
    return df


@pytest.fixture()
def source_sts():
    with moto.mock_sts():
        yield boto3.client("sts", region_name="us-east-2")


@pytest.fixture()
def source_s3():
    with moto.mock_s3():
        yield boto3.client("s3", region_name="us-east-2")


@pytest.fixture()
def source_iam():
    with moto.mock_iam():
        yield boto3.client("iam", region_name="us-east-2")


@pytest.fixture()
def vision_sts():
    with moto.mock_sts():
        yield boto3.client("sts", region_name="us-east-2")


@pytest.fixture()
def vision_s3():
    with moto.mock_s3():
        yield boto3.client("s3", region_name="us-east-2")


@pytest.fixture()
def vision_ec2():
    with moto.mock_ec2():
        yield boto3.client("ec2", region_name="us-east-2")


@pytest.fixture()
def vision_emr():
    with moto.mock_emr():
        yield boto3.client("emr", region_name="us-east-2")


@pytest.fixture()
def vision_iam():
    with moto.mock_iam():
        yield boto3.client("iam", region_name="us-east-2")


@pytest.fixture()
def divina_test_version():
    return pkg_resources.get_distribution("divina").version


@pytest.fixture()
def vision_codeartifact():
    return boto3.client("codeartifact", region_name="us-west-2")


@pytest.fixture()
def account_number(monkeypatch):
    monkeypatch.setenv("ACCOUNT_NUMBER", "123456789012")


@pytest.fixture()
def ec2_pricing_stub():
    return pricing_stubs.ec2_pricing_stub
