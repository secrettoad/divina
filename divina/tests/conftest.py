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
def random_state():
    return 11


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
def dask_client_remote(request):
    cluster = EC2Cluster(
        key_name="divina2",
        security=False,
        docker_image="jhurdle/divina:test",
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
    request.addfinalizer(lambda: client.close())
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
    train_df = test_df_1.groupby('a').agg('sum').reset_index()
    train_df['c'] = train_df['c'].shift(-1)
    train_df = train_df[train_df['a'] < "1970-01-01 00:00:07"]
    model = LinearRegression()
    model.fit(
        ddf.from_pandas(train_df, chunksize=10000)[["b"]].to_dask_array(lengths=True),
        ddf.from_pandas(train_df, chunksize=10000)["c"],
    )
    return model


@pytest.fixture()
def test_params_1(test_model_1):
    return {'params': {'b': test_model_1._coef[0]}}


@pytest.fixture()
def test_bootstrap_models(test_df_1, random_state):
    train_df = test_df_1.groupby('a').agg('sum').reset_index()
    train_df['c'] = train_df['c'].shift(-1)
    train_df = train_df[train_df['a'] < "1970-01-01 00:00:07"]
    bootstrap_models = {}
    for rs in range(random_state, random_state + 30):
        model = LinearRegression()
        confidence_df = train_df.sample(frac=.8, random_state=rs)
        model.fit(
            ddf.from_pandas(confidence_df, chunksize=10000)[["b"]].to_dask_array(lengths=True),
            ddf.from_pandas(confidence_df, chunksize=10000)["c"],
        )
        bootstrap_models[rs] = model
    return bootstrap_models


@pytest.fixture()
def test_params_2(test_model_1):
    return {'params': {'b': test_model_1._coef[0] + 1}}


@pytest.fixture()
def test_metrics_1():
    return {'splits': {'1970-01-01 00:00:07': {'time_horizons': {'1': {'mae': 11.496980676328501}}}}}


@pytest.fixture()
def test_val_predictions_1():
    df = pd.DataFrame(
        [[Timestamp('1970-01-01 00:00:01'), 30.465428743961354], [Timestamp('1970-01-01 00:00:04'), 6.537892512077306],
         [Timestamp('1970-01-01 00:00:05'), 16.507699275362324], [Timestamp('1970-01-01 00:00:06'), 36.44731280193237],
         [Timestamp('1970-01-01 00:00:07'), -49.293025362318815]]
    )
    df.columns = ["a", "c_h_1_pred"]
    return df


@pytest.fixture()
def test_forecast_1():
    df = pd.DataFrame(
        [[Timestamp('1970-01-01 00:00:05'), 16.507699275362324, 20.89668367346938],
         [Timestamp('1970-01-01 00:00:06'), 36.44731280193237, 41.55038265306125],
         [Timestamp('1970-01-01 00:00:07'), -49.293025362318815, -4.517509010484922],
         [Timestamp('1970-01-01 00:00:10'), 46.41711956521739, 58.77768987341773],
         [Timestamp('1970-01-01 00:00:10'), 46.41711956521739, 58.77768987341773],
         [Timestamp('1970-01-01 00:00:11'), 46.41711956521739, 58.77768987341773],
         [Timestamp('1970-01-01 00:00:12'), 46.41711956521739, 58.77768987341773],
         [Timestamp('1970-01-01 00:00:13'), 46.41711956521739, 58.77768987341773],
         [Timestamp('1970-01-01 00:00:14'), 46.41711956521739, 58.77768987341773],
         [Timestamp('1970-01-01 00:00:10'), 44.42315821256039, 55.03141952983726],
         [Timestamp('1970-01-01 00:00:10'), 44.42315821256039, 55.03141952983726],
         [Timestamp('1970-01-01 00:00:11'), 44.42315821256039, 55.03141952983726],
         [Timestamp('1970-01-01 00:00:12'), 44.42315821256039, 55.03141952983726],
         [Timestamp('1970-01-01 00:00:13'), 44.42315821256039, 55.03141952983726],
         [Timestamp('1970-01-01 00:00:14'), 44.42315821256039, 55.03141952983726],
         [Timestamp('1970-01-01 00:00:10'), 42.429196859903385, 51.28514918625679],
         [Timestamp('1970-01-01 00:00:10'), 42.429196859903385, 51.28514918625679],
         [Timestamp('1970-01-01 00:00:11'), 42.429196859903385, 51.28514918625679],
         [Timestamp('1970-01-01 00:00:12'), 42.429196859903385, 51.28514918625679],
         [Timestamp('1970-01-01 00:00:13'), 42.429196859903385, 51.28514918625679],
         [Timestamp('1970-01-01 00:00:14'), 42.429196859903385, 51.28514918625679],
         [Timestamp('1970-01-01 00:00:10'), 40.435235507246375, 47.53887884267631],
         [Timestamp('1970-01-01 00:00:10'), 40.435235507246375, 47.53887884267631],
         [Timestamp('1970-01-01 00:00:11'), 40.435235507246375, 47.53887884267631],
         [Timestamp('1970-01-01 00:00:12'), 40.435235507246375, 47.53887884267631],
         [Timestamp('1970-01-01 00:00:13'), 40.435235507246375, 47.53887884267631],
         [Timestamp('1970-01-01 00:00:14'), 40.435235507246375, 47.53887884267631],
         [Timestamp('1970-01-01 00:00:10'), 38.44127415458937, 43.88303571428574],
         [Timestamp('1970-01-01 00:00:10'), 38.44127415458937, 43.88303571428574],
         [Timestamp('1970-01-01 00:00:11'), 38.44127415458937, 43.88303571428574],
         [Timestamp('1970-01-01 00:00:12'), 38.44127415458937, 43.88303571428574],
         [Timestamp('1970-01-01 00:00:13'), 38.44127415458937, 43.88303571428574],
         [Timestamp('1970-01-01 00:00:14'), 38.44127415458937, 43.88303571428574],
         [Timestamp('1970-01-01 00:00:10'), 36.44731280193237, 41.55038265306125],
         [Timestamp('1970-01-01 00:00:10'), 36.44731280193237, 41.55038265306125],
         [Timestamp('1970-01-01 00:00:11'), 36.44731280193237, 41.55038265306125],
         [Timestamp('1970-01-01 00:00:12'), 36.44731280193237, 41.55038265306125],
         [Timestamp('1970-01-01 00:00:13'), 36.44731280193237, 41.55038265306125],
         [Timestamp('1970-01-01 00:00:14'), 36.44731280193237, 41.55038265306125]]
    )
    df.columns = ["a", "c_h_1_pred", "c_h_1_pred_c_90"]
    df.index = list(df.index + 1)
    df.index.name = 'new_index'
    return df


@pytest.fixture()
def test_fd_1():
    return {
        "forecast_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": ["1970-01-01 00:00:07"],
            "validate_start": "1970-01-01 00:00:01",
            "validate_end": "1970-01-01 00:00:09",
            "forecast_start": "1970-01-01 00:00:05",
            "forecast_end": "1970-01-01 00:00:14",
            "forecast_freq": 'S',
            "confidence_intervals": [90],
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
