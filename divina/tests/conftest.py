import json
import os
import pathlib
import shutil
from unittest.mock import patch

import boto3
import dask.dataframe as ddf
import fsspec
import numpy as np
import pandas as pd
import pytest
import s3fs
from dask.distributed import Client
from dask_cloudprovider.aws import EC2Cluster
from dask_ml.linear_model import LinearRegression
from pandas import Timestamp
from ..model import GLM


@pytest.fixture()
def test_bucket():
    return "s3://divina-test"


@pytest.fixture()
def random_state():
    return 11


def setup_teardown_test_bucket(s3_fs, test_bucket):
    fsspec.filesystem("s3").invalidate_cache()
    try:
        s3_fs.rm(test_bucket, recursive=True)
    except FileNotFoundError:
        pass
    s3_fs.mkdir(
        test_bucket,
        region_name=os.environ["AWS_DEFAULT_REGION"],
        acl="private",
    )
    yield
    try:
        s3_fs.rm(test_bucket, recursive=True)
    except FileNotFoundError:
        pass


@pytest.fixture()
def setup_teardown_test_bucket_contents(s3_fs, test_bucket):
    fsspec.filesystem("s3").invalidate_cache()
    try:
        s3_fs.rm(test_bucket, recursive=True)
    except FileNotFoundError:
        pass
    s3_fs.mkdir(
        test_bucket,
        region_name=os.environ["AWS_DEFAULT_REGION"],
        acl="private",
    )
    yield
    try:
        s3_fs.rm(test_bucket, recursive=True)
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


@pytest.fixture(scope="session")
def s3_fs():
    return s3fs.S3FileSystem()


@pytest.fixture(scope="session")
def dask_cluster_ip():
    return None


@pytest.fixture(scope="session")
def dask_client_remote(request, dask_cluster_ip):
    if dask_cluster_ip:
        client = Client(dask_cluster_ip)
        yield client
    else:
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
        cluster.scale(5)
        client = Client(cluster)
        request.addfinalizer(lambda: client.close())
        yield client
        if not dask_cluster_ip:
            client.shutdown()


@pytest.fixture()
def test_model_1(test_df_1, random_state, test_ed_1):
    params = [2.78841111,  0.93118321, -2.24143098]
    intercept = 2.2413622182946873
    features = ["b", "b_(5, 10]", "b_(15, inf]"]

    model = GLM(link_function='log')
    model.linear_model.fit(
        ddf.from_pandas(
            pd.DataFrame([np.array(params) + c for c in range(0, 2)]), npartitions=1
        ).to_dask_array(lengths=True),
        ddf.from_pandas(pd.Series([intercept, intercept]), npartitions=1).to_dask_array(
            lengths=True
        ),
    )
    model.linear_model.coef_ = np.array(params)
    model.linear_model.intercept_ = intercept
    model.linear_model._coef = np.array(params + [intercept])
    return (model, {"features": features})


@pytest.fixture()
def test_params_1(test_model_1):
    return test_model_1[1]


@pytest.fixture()
def test_bootstrap_models(test_df_1, random_state, test_ed_1):
    params = [
        [3.5208353651349817, 0.8890968752323547, -3.3105628218449574],
        [1.1717451589768473, 0.9272367214532506, -2.972442774585969],
        [2.5578331046564813, 0.9201254489380151, -2.0817764266201166],
        [2.557833104656474, 0.9201254489380154, -2.081776426620107],
        [2.514451997020174, 0.930893974845844, -1.406470131803517],
    ]
    intercepts = [
        3.310380317261251,
        2.9723833353906963,
        2.0817011992723984,
        2.0817011992723877,
        1.4064614399090596,
    ]
    features = [
        ["b", "b_(5, 10]", "b_(15, inf]"],
        ["b", "b_(5, 10]", "b_(15, inf]"],
        ["b", "b_(5, 10]", "b_(15, inf]"],
        ["b", "b_(5, 10]", "b_(15, inf]"],
        ["b", "b_(5, 10]", "b_(15, inf]"],
    ]
    states = range(
        random_state,
        random_state + test_ed_1["experiment_definition"]["bootstrap_sample"],
    )
    bootstrap_models = {}

    for j, i, p, f, state in zip(
        range(0, len(states)), intercepts, params, features, states
    ):
        model = LinearRegression()
        model.fit(
            ddf.from_pandas(
                pd.DataFrame([np.array(params[j]) + c for c in range(0, len(states))]),
                npartitions=1,
            ).to_dask_array(lengths=True),
            ddf.from_pandas(pd.Series(intercepts), npartitions=1).to_dask_array(
                lengths=True
            ),
        )
        model.coef_ = np.array(p)
        model.intercept_ = i
        model._coef = np.array(p + [i])
        bootstrap_models[state] = (model, {"features": f})
    return bootstrap_models


@pytest.fixture()
def test_validation_models(test_df_1, random_state, test_ed_1):
    params = [[4.086805634357628, 0.7618639411983616, -1.6591807041382545]]
    intercepts = [1.6591672730002625]
    features = [["b", "b_(5, 10]", "b_(15, inf]"]]
    splits = test_ed_1["experiment_definition"]["validation_splits"]
    validation_models = {}

    for j, i, p, f, split in zip(
        range(0, len(splits)), intercepts, params, features, splits
    ):
        model = LinearRegression()
        model.fit(
            ddf.from_pandas(
                pd.DataFrame(
                    [np.array(params[j]) + c for c in range(1, len(splits) + 1)]
                    + [np.array(params[j])]
                ),
                npartitions=1,
            ).to_dask_array(lengths=True),
            ddf.from_pandas(pd.Series(intercepts + [i]), npartitions=1).to_dask_array(
                lengths=True
            ),
        )
        model.coef_ = np.array(p)
        model.intercept_ = i
        model._coef = np.array(p + [i])
        validation_models[split] = (model, {"features": f})
    return validation_models


@pytest.fixture()
def test_params_2(test_model_1):
    return test_model_1[1]


@pytest.fixture()
def test_metrics_1():
    return {'mse': 7.403630461048873e+65}


@pytest.fixture()
def test_val_predictions_1():
    df = pd.DataFrame(
        [
            [Timestamp("1970-01-01 00:00:01"), 8.522536459806267],
            [Timestamp("1970-01-01 00:00:04"), 20.983251731325122],
            [Timestamp("1970-01-01 00:00:05"), 17.173932025333315],
            [Timestamp("1970-01-01 00:00:06"), 6.236944636211182],
            [Timestamp("1970-01-01 00:00:07"), 42.31544208487925],
        ]
    )
    df.columns = ["a", "c_h_1_pred"]
    return df


@pytest.fixture()
def test_forecast_1():
    return ddf.from_array(np.array([2970.3819018152676, 18730846144.84451, 178040476.4682159, 181.7979998940561, 3.9443353279256965e+21, 2.1076475693600494e+33])).to_frame()


@pytest.fixture()
def test_ed_1():
    return {
        "experiment_definition": {
            "time_index": "a",
            "target": "c",
            "validation_splits": ["1970-01-01 00:00:07"],
            "validate_start": "1970-01-01 00:00:01",
            "validate_end": "1970-01-01 00:00:09",
            "forecast_start": "1970-01-01 00:00:05",
            "forecast_end": "1970-01-01 00:00:14",
            "frequency": "S",
            "confidence_intervals": [90],
            "bootstrap_sample": 5,
            "bin_features": {"b": [5, 10, 15]},
            "scenarios": {
                "b": {"mode": "constant", "constant_values": [0, 1, 2, 3, 4, 5]}
            },
            "time_horizons": [1],
            #"data_path": "divina-test/dataset/test1",
        }
    }

@pytest.fixture()
def test_ed_2():
    return {
        "experiment_definition": {
            "time_index": "a",
            "target": "c",
            "validation_splits": ["1970-01-01 00:00:07"],
            "validate_start": "1970-01-01 00:00:01",
            "validate_end": "1970-01-01 00:00:09",
            "forecast_start": "1970-01-01 00:00:05",
            "forecast_end": "1970-01-01 00:00:14",
            "frequency": "S",
            "confidence_intervals": [90],
            "bootstrap_sample": 5,
            "bin_features": {"b": [5, 10, 15]},
            "scenarios": {
                "b": {"mode": "constant", "constant_values": [0, 1, 2, 3, 4, 5]}
            },
            "time_horizons": [1],
            "target_dimensions": ['d', 'e']
            #"data_path": "divina-test/dataset/test1",
        }
    }


@pytest.fixture()
def test_eds_quickstart():
    eds = {}
    for file in sorted(
        os.listdir(
            pathlib.Path(
                pathlib.Path(__file__).parent.parent.parent,
                "docs_src/_static/experiment_definitions",
            )
        )
    ):
        with open(
            pathlib.Path(
                pathlib.Path(__file__).parent.parent.parent,
                "docs_src/_static/experiment_definitions",
                file,
            )
        ) as f:
            eds[file.split(".")[0]] = json.load(f)
    return eds


@pytest.fixture()
def test_ed_3(test_bucket, test_ed_1):
    test_ed = test_ed_1
    #test_ed["experiment_definition"].update(
    #    {"data_path": "{}/dataset/test1".format(test_bucket)}
    #)
    return test_ed


@pytest.fixture()
def test_composite_dataset_1():
    df = pd.DataFrame(
        [
            [Timestamp("1970-01-01 00:00:01"), 8.0, 12.0, 2.0, 3.0],
            [Timestamp("1970-01-01 00:00:04"), 20.0, 24.0, np.NaN, 6.0],
            [Timestamp("1970-01-01 00:00:05"), 15.0, 18.0, np.NaN, np.NaN],
            [Timestamp("1970-01-01 00:00:06"), 5.0, 6.0, np.NaN, np.NaN],
            [Timestamp("1970-01-01 00:00:07"), 48.0, 54.0, 8.0, np.NaN],
            [Timestamp("1970-01-01 00:00:10"), 77.0, 84.0, np.NaN, np.NaN],
        ]
    )
    df.columns = ["a", "b", "c", "e", "f"]
    return df


@pytest.fixture()
def test_data_1():
    df = (
        pd.DataFrame(
            [
                [Timestamp("1970-01-01 00:00:01"), 2.0, 3.0],
                [Timestamp("1970-01-01 00:00:04"), 5.0, 6.0],
                [Timestamp("1970-01-01 00:00:05"), 5.0, 6.0],
                [Timestamp("1970-01-01 00:00:06"), 5.0, 6.0],
                [Timestamp("1970-01-01 00:00:07"), 8.0, 9],
                [Timestamp("1970-01-01 00:00:10"), 11.0, 12.0],
            ]
        )
        .sample(25, replace=True, random_state=11)
        .reset_index(drop=True)
    )
    df.columns = ["a", "b", "c"]
    return  ddf.from_pandas(df, npartitions=2)


@pytest.fixture()
def test_df_1():
    df = pd.DataFrame(
         [[Timestamp('1970-01-01 00:00:01'), 8.0, 12.0, 1, 0],
                    [Timestamp('1970-01-01 00:00:04'), 20.0, 24.0, 0, 1],
                    [Timestamp('1970-01-01 00:00:05'), 15.0, 18.0, 0, 1],
                    [Timestamp('1970-01-01 00:00:06'), 5.0, 6.0, 1, 0],
                    [Timestamp('1970-01-01 00:00:07'), 48.0, 54.0, 0, 1],
                    [Timestamp('1970-01-01 00:00:10'), 77.0, 84.0, 0, 1]]
    )
    df.columns = ['a', 'b', 'c', 'b_(5, 10]', 'b_(15, inf]']
    return ddf.from_pandas(df, npartitions=2)


@pytest.fixture()
def test_df_2():
    df = pd.DataFrame(
        [
            [Timestamp("1970-01-01 00:00:01"), 2.0, 3.0],
            [Timestamp("1970-01-01 00:00:04"), np.NaN, 6.0],
            [Timestamp("1970-01-01 00:00:07"), 8.0, np.NaN],
            [np.NaN, 11.0, 12.0],
        ]
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
def test_df_4():
    df = pd.DataFrame(
         [[Timestamp('1970-01-01 00:00:01'), 8.0, 12.0, 1, 0],
                    [Timestamp('1970-01-01 00:00:04'), 20.0, 24.0, 0, 1],
                    [Timestamp('1970-01-01 00:00:05'), 15.0, 18.0, 0, 1],
                    [Timestamp('1970-01-01 00:00:06'), 5.0, 6.0, 1, 0],
                    [Timestamp('1970-01-01 00:00:07'), 48.0, 54.0, 0, 1],
                    [Timestamp('1970-01-01 00:00:10'), 77.0, 84.0, 0, 1],
          [Timestamp('1970-01-02 00:00:01'), 8.0, 12.0, 1, 0],
          [Timestamp('1970-01-02 00:00:04'), 20.0, 24.0, 0, 1],
          [Timestamp('1970-01-02 00:00:05'), 15.0, 18.0, 0, 1],
          [Timestamp('1970-01-02 00:00:06'), 5.0, 6.0, 1, 0],
          [Timestamp('1970-01-02 00:00:07'), 48.0, 54.0, 0, 1],
          [Timestamp('1970-01-02 00:00:10'), 77.0, 84.0, 0, 1],

          [Timestamp('1970-01-03 00:00:01'), 8.0, 12.0, 1, 0],
          [Timestamp('1970-01-03 00:00:04'), 20.0, 24.0, 0, 1],
          [Timestamp('1970-01-03 00:00:05'), 15.0, 18.0, 0, 1],
          [Timestamp('1970-01-03 00:00:06'), 5.0, 6.0, 1, 0],
          [Timestamp('1970-01-03 00:00:07'), 48.0, 54.0, 0, 1],
          [Timestamp('1970-01-03 00:00:10'), 77.0, 84.0, 0, 1],

          [Timestamp('1970-01-04 00:00:01'), 8.0, 12.0, 1, 0],
          [Timestamp('1970-01-04 00:00:04'), 20.0, 24.0, 0, 1],
          [Timestamp('1970-01-04 00:00:05'), 15.0, 18.0, 0, 1],
          [Timestamp('1970-01-04 00:00:06'), 5.0, 6.0, 1, 0],
          [Timestamp('1970-01-04 00:00:07'), 48.0, 54.0, 0, 1],
          [Timestamp('1970-01-04 00:00:10'), 77.0, 84.0, 0, 1],

          [Timestamp('1970-01-05 00:00:01'), 8.0, 12.0, 1, 0],
          [Timestamp('1970-01-05 00:00:04'), 20.0, 24.0, 0, 1],
          [Timestamp('1970-01-05 00:00:05'), 15.0, 18.0, 0, 1],
          [Timestamp('1970-01-05 00:00:06'), 5.0, 6.0, 1, 0],
          [Timestamp('1970-01-05 00:00:07'), 48.0, 54.0, 0, 1],
          [Timestamp('1970-01-05 00:00:10'), 77.0, 84.0, 0, 1]
          ]
    )
    df.columns = ['a', 'b', 'c', 'b_(5, 10]', 'b_(15, inf]']
    for c in ['e', 'd']:
        df[c] = np.random.randint(1, 3, df.shape[0])
    return ddf.from_pandas(df, npartitions=2)
