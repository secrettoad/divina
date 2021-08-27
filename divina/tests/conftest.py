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
import fsspec


'''@pytest.fixture(autouse=True, scope='session')
def setup_teardown_test_bucket(s3_fs):
    try:
        s3_fs.mkdir(
            os.environ["TEST_BUCKET"],
            region_name=os.environ["AWS_DEFAULT_REGION"],
            acl="private",
        )
    except FileExistsError:
        s3_fs.rm(os.environ["TEST_BUCKET"], recursive=True)
        s3_fs.mkdir(
            os.environ["TEST_BUCKET"],
            region_name=os.environ["AWS_DEFAULT_REGION"],
            acl="private",
        )
    try:
        os.mkdir("divina-test")
    except FileExistsError:
        shutil.rmtree("divina-test")
        os.mkdir("divina-test")
    fsspec.filesystem("s3").invalidate_cache()
    yield
    try:
        s3_fs.rm(os.environ["TEST_BUCKET"], recursive=True)
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree("divina-test")
    except FileNotFoundError:
        pass'''


@pytest.fixture(autouse=True)
def setup_teardown_test_bucket_contents(s3_fs, request):
    test_path = '{}/{}'.format(os.environ['TEST_BUCKET'], request.node.originalname)
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
    try:
        os.mkdir("divina-test")
    except FileExistsError:
        shutil.rmtree("divina-test")
        os.mkdir("divina-test")
    fsspec.filesystem("s3").invalidate_cache()
    yield
    try:
        s3_fs.rm(test_path, recursive=True)
    except FileNotFoundError:
        pass
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


@pytest.fixture(scope='session')
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
    )
    cluster.adapt(minimum=1, maximum=10)
    client = Client(cluster)
    yield client
    client.close()


@pytest.fixture()
def vd_no_dataset_id():
    return {
        "vision_definition": {
            "target": "c",
            "time_index": "test1",
            "dataset_id": "test1",
        }
    }


@pytest.fixture()
def vd_no_dataset_directory():
    return {
        "vision_definition": {
            "target": "c",
            "time_index": "test1",
            "dataset_directory": "divina-test/dataset",
        }
    }


@pytest.fixture()
def vd_no_time_index():
    return {
        "vision_definition": {
            "target": "c",
            "dataset_directory": "divina-test/dataset",
            "dataset_id": "test1",
        }
    }


@pytest.fixture()
def vd_no_target():
    return {
        "vision_definition": {
            "time_index": "a",
            "dataset_directory": "divina-test/dataset",
            "dataset_id": "test1",
        }
    }


@pytest.fixture()
def vd_time_validation_splits_not_list():
    return {
        "vision_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": "1970-01-01 00:00:08",
            "time_horizons": [1],
            "dataset_directory": "divina-test/dataset",
            "dataset_id": "test1",
        }
    }


@pytest.fixture()
def vd_time_horizons_not_list():
    return {
        "vision_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": 1,
            "dataset_directory": "divina-test/dataset",
            "dataset_id": "test1",
        }
    }


@pytest.fixture()
def test_model_1(test_df_1):
    return LinearRegression().fit(
        ddf.from_pandas(test_df_1, chunksize=10000)[["b"]].to_dask_array(lengths=True),
        ddf.from_pandas(test_df_1, chunksize=10000)["c"],
    )


@pytest.fixture()
def test_metrics_1():
    return {
        "splits": {
            "1970-01-01 00:00:08": {"time_horizons": {"1": {"mae": 0.3750750300119954}}}
        }
    }


@pytest.fixture()
def test_predictions_1():
    df = pd.DataFrame(
        [
            [4.0, 5.836397058823538],
            [1.0, 2.9181985294117725],
            [6.0, 5.836397058823538],
            [4.0, 5.836397058823538],
            [10.0, 11.672794117647069],
            [7.0, 8.754595588235304],
            [4.0, 5.836397058823538],
            [5.0, 5.836397058823538],
            [1.0, 2.9181985294117725],
            [10.0, 11.672794117647069],
            [7.0, 8.754595588235304],
            [1.0, 2.9181985294117725],
            [10.0, 11.672794117647069],
            [1.0, 2.9181985294117725],
            [10.0, 11.672794117647069],
            [7.0, 8.754595588235304],
            [10.0, 11.672794117647069],
            [7.0, 8.754595588235304],
            [5.0, 5.836397058823538],
            [7.0, 8.754595588235304],
            [4.0, 5.836397058823538],
            [5.0, 5.836397058823538],
            [10.0, 11.672794117647069],
            [10.0, 11.672794117647069],
            [7.0, 8.754595588235304],
        ]
    )
    df.columns = ["a", "c_h_1_pred"]
    return df


@pytest.fixture()
def test_vd_1():
    return {
        "vision_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": [1],
            "dataset_directory": "divina-test/dataset",
            "dataset_id": "test1",
        }
    }


@pytest.fixture()
def test_vd_2():
    return {
        "vision_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": [1],
            "dataset_directory": "divina-test/dataset",
            "dataset_id": "test1",
            "joins": [
                {
                    "dataset_directory": "dataset",
                    "dataset_id": "test2",
                    "join_on": ("a", "test2_a"),
                }
            ],
        }
    }


@pytest.fixture()
def test_vd_3():
    return {
        "vision_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": [1],
            "dataset_directory": "{}/dataset".format(os.environ["TEST_BUCKET"]),
            "dataset_id": "test1",
        }
    }


@pytest.fixture()
def test_composite_dataset_1():
    df = pd.DataFrame(
        [
            [4.0, 5.0, 6.0, 4.0, np.NaN, 6.0],
            [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            [6.0, 5.0, 6.0, np.NaN, np.NaN, np.NaN],
            [4.0, 5.0, 6.0, 4.0, np.NaN, 6.0],
            [10.0, 11.0, 12.0, np.NaN, np.NaN, np.NaN],
            [7.0, 8.0, 9.0, 7.0, 8.0, np.NaN],
            [4.0, 5.0, 6.0, 4.0, np.NaN, 6.0],
            [5.0, 5.0, 6.0, np.NaN, np.NaN, np.NaN],
            [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0, np.NaN, np.NaN, np.NaN],
            [7.0, 8.0, 9.0, 7.0, 8.0, np.NaN],
            [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0, np.NaN, np.NaN, np.NaN],
            [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0, np.NaN, np.NaN, np.NaN],
            [7.0, 8.0, 9.0, 7.0, 8.0, np.NaN],
            [10.0, 11.0, 12.0, np.NaN, np.NaN, np.NaN],
            [7.0, 8.0, 9.0, 7.0, 8.0, np.NaN],
            [5.0, 5.0, 6.0, np.NaN, np.NaN, np.NaN],
            [7.0, 8.0, 9.0, 7.0, 8.0, np.NaN],
            [4.0, 5.0, 6.0, 4.0, np.NaN, 6.0],
            [5.0, 5.0, 6.0, np.NaN, np.NaN, np.NaN],
            [10.0, 11.0, 12.0, np.NaN, np.NaN, np.NaN],
            [10.0, 11.0, 12.0, np.NaN, np.NaN, np.NaN],
            [7.0, 8.0, 9.0, 7.0, 8.0, np.NaN],
        ]
    )
    df.columns = ["a", "b", "c", "test2_a", "test2_e", "test2_f"]
    return df


@pytest.fixture()
def test_composite_profile_1():
    df = pd.DataFrame(
        [
            ["25%", 4.0, 5.0, 6.0, 2.5, 5.0, 4.5],
            ["50%", 7.0, 8.0, 9.0, 4.0, 8.0, 6.0],
            ["75%", 10.0, 11.0, 12.0, 5.5, 9.5, 9.0],
            ["count", 25.0, 25.0, 25.0, 3.0, 3.0, 3.0],
            ["max", 10.0, 11.0, 12.0, 7.0, 11.0, 12.0],
            ["mean", 6.12, 6.92, 7.92, 4.0, 7.0, 7.0],
            ["min", 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            [
                "std",
                3.1400636936215163,
                3.2264531609803355,
                3.2264531609803355,
                3.0,
                4.58257569495584,
                4.58257569495584,
            ],
        ]
    )
    df.columns = ["statistic", "a", "b", "c", "test2_a", "test2_e", "test2_f"]
    df.set_index("statistic", inplace=True)
    df.index.name = None
    return df


@pytest.fixture()
def test_df_1():
    df = (
        pd.DataFrame(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [5.0, 5.0, 6.0],
                [6.0, 5.0, 6.0],
                [7.0, 8.0, 9],
                [10.0, 11.0, 12.0],
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
        [[1, 2.0, 3.0], [4, np.NaN, 6.0], [7, 8.0, np.NaN], [np.NaN, 11.0, 12.0]]
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
