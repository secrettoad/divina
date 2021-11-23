import pytest
import pandas as pd
import numpy as np
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
import json
import pathlib


@pytest.fixture()
def test_bucket():
    return "s3://divina-test-2"


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
def dask_client(request):
    client = Client()
    request.addfinalizer(lambda: client.close())
    yield client
    client.shutdown()


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
def fd_no_data_path():
    return {
        "experiment_definition": {
            "target": "c",
            "time_index": "a",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": [1],
        }
    }


@pytest.fixture()
def fd_invalid_model():
    return {
        "experiment_definition": {
            "target": "c",
            "time_index": "a",
            "data_path": "divina-test/dataset/test1",
            "model": "scikitlearn.linear_models.linearRegression",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": [1],
        }
    }


@pytest.fixture()
def fd_no_time_index():
    return {
        "experiment_definition": {
            "target": "c",
            "data_path": "divina-test/dataset/test1",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": [1],
        }
    }


@pytest.fixture()
def fd_no_target():
    return {
        "experiment_definition": {
            "time_index": "a",
            "data_path": "divina-test/dataset/test1",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": [1],
        }
    }


@pytest.fixture()
def fd_time_validation_splits_not_list():
    return {
        "experiment_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": "1970-01-01 00:00:08",
            "time_horizons": [1],
            "data_path": "divina-test/dataset/test1",
        }
    }


@pytest.fixture()
def fd_time_horizons_not_list():
    return {
        "experiment_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": 1,
            "data_path": "divina-test/dataset/test1",
        }
    }


@pytest.fixture()
def fd_time_horizons_range_not_tuple():
    return {
        "experiment_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": [[1, 60]],
            "data_path": "divina-test/dataset/test1",
        }
    }


@pytest.fixture()
def test_model_1(test_df_1, random_state, test_fd_1):
    params = [2.5980497162871465, 0.9334212709843847, -2.13558920798334]
    intercept = 2.135585380640592
    features = ['b', 'b_(5, 10]', 'b_(15, inf]']

    model = LinearRegression()
    model.fit(
        ddf.from_pandas(pd.DataFrame([np.array(params) + c for c in range(0, 2)]), npartitions=1).to_dask_array(
            lengths=True),
        ddf.from_pandas(pd.Series([intercept, intercept]), npartitions=1).to_dask_array(lengths=True))
    model.coef_ = np.array(params)
    model.intercept_ = intercept
    model._coef = np.array(params + [intercept])
    return (model, {"features": features})


@pytest.fixture()
def test_params_1(test_model_1):
    return test_model_1[1]


@pytest.fixture()
def test_bootstrap_models(test_df_1, random_state, test_fd_1):
    params = [[3.5208353651349817, 0.8890968752323547, -3.3105628218449574],
              [1.1717451589768473, 0.9272367214532506, -2.972442774585969],
              [2.5578331046564813, 0.9201254489380151, -2.0817764266201166],
              [2.557833104656474, 0.9201254489380154, -2.081776426620107],
              [2.514451997020174, 0.930893974845844, -1.406470131803517]]
    intercepts = [3.310380317261251, 2.9723833353906963, 2.0817011992723984, 2.0817011992723877, 1.4064614399090596]
    features = [['b', 'b_(5, 10]', 'b_(15, inf]'], ['b', 'b_(5, 10]', 'b_(15, inf]'], ['b', 'b_(5, 10]', 'b_(15, inf]'],
                ['b', 'b_(5, 10]', 'b_(15, inf]'], ['b', 'b_(5, 10]', 'b_(15, inf]']]
    seeds = range(random_state, random_state + test_fd_1["experiment_definition"]["bootstrap_sample"])
    bootstrap_models = {}

    for j, i, p, f, seed in zip(range(0, len(seeds)), intercepts, params, features, seeds):
        model = LinearRegression()
        model.fit(
            ddf.from_pandas(pd.DataFrame([np.array(params[j]) + c for c in range(0, len(seeds))]),
                            npartitions=1).to_dask_array(
                lengths=True),
            ddf.from_pandas(pd.Series(intercepts), npartitions=1).to_dask_array(lengths=True))
        model.coef_ = np.array(p)
        model.intercept_ = i
        model._coef = np.array(p + [i])
        bootstrap_models[seed] = (model, {"features": f})
    return bootstrap_models


@pytest.fixture()
def test_validation_models(test_df_1, random_state, test_fd_1):
    params = [[4.086805634357628, 0.7618639411983616, -1.6591807041382545]]
    intercepts = [1.6591672730002625]
    features = [['b', 'b_(5, 10]', 'b_(15, inf]']]
    splits = test_fd_1["experiment_definition"]["time_validation_splits"]
    validation_models = {}

    for j, i, p, f, split in zip(range(0, len(splits)), intercepts, params, features, splits):
        model = LinearRegression()
        model.fit(
            ddf.from_pandas(
                pd.DataFrame([np.array(params[j]) + c for c in range(1, len(splits) + 1)] + [np.array(params[j])]),
                npartitions=1).to_dask_array(
                lengths=True),
            ddf.from_pandas(pd.Series(intercepts + [i]), npartitions=1).to_dask_array(lengths=True))
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
    return {'splits': {'1970-01-01 00:00:07': {'time_horizons': {'1': {'mae': 19.349425665160247}}}}}


@pytest.fixture()
def test_val_predictions_1():
    df = pd.DataFrame(
        [[Timestamp('1970-01-01 00:00:01'), 8.522536459806267], [Timestamp('1970-01-01 00:00:04'), 20.983251731325122],
         [Timestamp('1970-01-01 00:00:05'), 17.173932025333315], [Timestamp('1970-01-01 00:00:06'), 6.236944636211182],
         [Timestamp('1970-01-01 00:00:07'), 42.31544208487925]]
    )
    df.columns = ['a', 'c_h_1_pred']
    return df


@pytest.fixture()
def test_forecast_1():
    df = pd.DataFrame(
        [[Timestamp('1970-01-01 00:00:05'), 15.0, 18.0, 0, 0, 0, 1, 18.73495416169351, 38.970745744307195, 0.0,
          -2.13558920798334, 20.167668810881555, 18.0526793161663, 18.441416037999105, 18.44141603799909,
          17.884323059616893, 19.45131148628753],
         [Timestamp('1970-01-01 00:00:06'), 5.0, 6.0, 0, 1, 0, 0, 5.12956686322573, 12.990248581435733,
          0.9334212709843847, -0.0, 4.655756919451798, 2.8354859916571313, 5.07668392272644, 5.076683922726445,
          5.762451739445877, 5.4460093013358035],
         [Timestamp('1970-01-01 00:00:07'), 48.0, 54.0, 0, 0, 0, 1, 49.5378561041782, 124.70638638178303, 0.0,
          -2.13558920798334, 49.50786569354926, 48.651491124123574, 48.8055558529536, 48.8055558529536,
          48.60382422952974, 49.52286089886373],
         [Timestamp('1970-01-01 00:00:10'), 77.0, 84.0, 0, 0, 0, 1, 76.60707296272535, 200.0498281541103, 0.0,
          -2.13558920798334, 75.29167507528754, 75.54135604626785, 75.48919387215604, 75.48919387215605,
          75.59974950005923, 76.1034112313923],
         [Timestamp('1970-01-01 00:00:11'), 0.0, np.NaN, 1, 0, 0, 0, 2.5980497162871465, 0.0, 0.0, -0.0,
          3.5208353651349817, 1.1717451589768473, 2.5578331046564813, 2.557833104656474, 2.514451997020174,
          3.059442540711064],
         [Timestamp('1970-01-01 00:00:11'), 1.0, np.NaN, 1, 0, 0, 0, 3.5314709872715313, 2.5980497162871465, 0.0, -0.0,
          4.409932240367336, 2.098981880430098, 3.4779585535944966, 3.4779585535944895, 3.4453459718660184,
          3.9707016138194335],
         [Timestamp('1970-01-01 00:00:11'), 2.0, np.NaN, 1, 0, 0, 0, 4.464892258255916, 5.196099432574293, 0.0, -0.0,
          5.299029115599691, 3.0262186018833486, 4.398084002532512, 4.398084002532505, 4.376239946711863,
          4.881960686927803],
         [Timestamp('1970-01-01 00:00:11'), 3.0, np.NaN, 1, 0, 0, 0, 5.3983135292403, 7.79414914886144, 0.0, -0.0,
          6.188125990832046, 3.953455323336599, 5.318209451470526, 5.318209451470521, 5.307133921557706,
          5.793219760036173],
         [Timestamp('1970-01-01 00:00:11'), 4.0, np.NaN, 1, 0, 0, 0, 6.331734800224686, 10.392198865148586, 0.0, -0.0,
          7.0772228660644005, 4.88069204478985, 6.2383349004085416, 6.238334900408535, 6.23802789640355,
          6.7044788331445435],
         [Timestamp('1970-01-01 00:00:11'), 5.0, np.NaN, 0, 1, 0, 0, 5.12956686322573, 12.990248581435733,
          0.9334212709843847, -0.0, 4.655756919451798, 2.8354859916571313, 5.07668392272644, 5.076683922726445,
          5.762451739445877, 5.4460093013358035],
         [Timestamp('1970-01-01 00:00:12'), 5.0, np.NaN, 0, 1, 0, 0, 5.12956686322573, 12.990248581435733,
          0.9334212709843847, -0.0, 4.655756919451798, 2.8354859916571313, 5.07668392272644, 5.076683922726445,
          5.762451739445877, 5.4460093013358035],
         [Timestamp('1970-01-01 00:00:12'), 4.0, np.NaN, 1, 0, 0, 0, 6.331734800224686, 10.392198865148586, 0.0, -0.0,
          7.0772228660644005, 4.88069204478985, 6.2383349004085416, 6.238334900408535, 6.23802789640355,
          6.7044788331445435],
         [Timestamp('1970-01-01 00:00:12'), 3.0, np.NaN, 1, 0, 0, 0, 5.3983135292403, 7.79414914886144, 0.0, -0.0,
          6.188125990832046, 3.953455323336599, 5.318209451470526, 5.318209451470521, 5.307133921557706,
          5.793219760036173],
         [Timestamp('1970-01-01 00:00:12'), 1.0, np.NaN, 1, 0, 0, 0, 3.5314709872715313, 2.5980497162871465, 0.0, -0.0,
          4.409932240367336, 2.098981880430098, 3.4779585535944966, 3.4779585535944895, 3.4453459718660184,
          3.9707016138194335],
         [Timestamp('1970-01-01 00:00:12'), 0.0, np.NaN, 1, 0, 0, 0, 2.5980497162871465, 0.0, 0.0, -0.0,
          3.5208353651349817, 1.1717451589768473, 2.5578331046564813, 2.557833104656474, 2.514451997020174,
          3.059442540711064],
         [Timestamp('1970-01-01 00:00:12'), 2.0, np.NaN, 1, 0, 0, 0, 4.464892258255916, 5.196099432574293, 0.0, -0.0,
          5.299029115599691, 3.0262186018833486, 4.398084002532512, 4.398084002532505, 4.376239946711863,
          4.881960686927803],
         [Timestamp('1970-01-01 00:00:13'), 0.0, np.NaN, 1, 0, 0, 0, 2.5980497162871465, 0.0, 0.0, -0.0,
          3.5208353651349817, 1.1717451589768473, 2.5578331046564813, 2.557833104656474, 2.514451997020174,
          3.059442540711064],
         [Timestamp('1970-01-01 00:00:13'), 1.0, np.NaN, 1, 0, 0, 0, 3.5314709872715313, 2.5980497162871465, 0.0, -0.0,
          4.409932240367336, 2.098981880430098, 3.4779585535944966, 3.4779585535944895, 3.4453459718660184,
          3.9707016138194335],
         [Timestamp('1970-01-01 00:00:13'), 2.0, np.NaN, 1, 0, 0, 0, 4.464892258255916, 5.196099432574293, 0.0, -0.0,
          5.299029115599691, 3.0262186018833486, 4.398084002532512, 4.398084002532505, 4.376239946711863,
          4.881960686927803],
         [Timestamp('1970-01-01 00:00:13'), 3.0, np.NaN, 1, 0, 0, 0, 5.3983135292403, 7.79414914886144, 0.0, -0.0,
          6.188125990832046, 3.953455323336599, 5.318209451470526, 5.318209451470521, 5.307133921557706,
          5.793219760036173],
         [Timestamp('1970-01-01 00:00:13'), 4.0, np.NaN, 1, 0, 0, 0, 6.331734800224686, 10.392198865148586, 0.0, -0.0,
          7.0772228660644005, 4.88069204478985, 6.2383349004085416, 6.238334900408535, 6.23802789640355,
          6.7044788331445435],
         [Timestamp('1970-01-01 00:00:13'), 5.0, np.NaN, 0, 1, 0, 0, 5.12956686322573, 12.990248581435733,
          0.9334212709843847, -0.0, 4.655756919451798, 2.8354859916571313, 5.07668392272644, 5.076683922726445,
          5.762451739445877, 5.4460093013358035],
         [Timestamp('1970-01-01 00:00:14'), 4.0, np.NaN, 1, 0, 0, 0, 6.331734800224686, 10.392198865148586, 0.0, -0.0,
          7.0772228660644005, 4.88069204478985, 6.2383349004085416, 6.238334900408535, 6.23802789640355,
          6.7044788331445435],
         [Timestamp('1970-01-01 00:00:14'), 0.0, np.NaN, 1, 0, 0, 0, 2.5980497162871465, 0.0, 0.0, -0.0,
          3.5208353651349817, 1.1717451589768473, 2.5578331046564813, 2.557833104656474, 2.514451997020174,
          3.059442540711064],
         [Timestamp('1970-01-01 00:00:14'), 1.0, np.NaN, 1, 0, 0, 0, 3.5314709872715313, 2.5980497162871465, 0.0, -0.0,
          4.409932240367336, 2.098981880430098, 3.4779585535944966, 3.4779585535944895, 3.4453459718660184,
          3.9707016138194335],
         [Timestamp('1970-01-01 00:00:14'), 2.0, np.NaN, 1, 0, 0, 0, 4.464892258255916, 5.196099432574293, 0.0, -0.0,
          5.299029115599691, 3.0262186018833486, 4.398084002532512, 4.398084002532505, 4.376239946711863,
          4.881960686927803],
         [Timestamp('1970-01-01 00:00:14'), 3.0, np.NaN, 1, 0, 0, 0, 5.3983135292403, 7.79414914886144, 0.0, -0.0,
          6.188125990832046, 3.953455323336599, 5.318209451470526, 5.318209451470521, 5.307133921557706,
          5.793219760036173],
         [Timestamp('1970-01-01 00:00:14'), 5.0, np.NaN, 0, 1, 0, 0, 5.12956686322573, 12.990248581435733,
          0.9334212709843847, -0.0, 4.655756919451798, 2.8354859916571313, 5.07668392272644, 5.076683922726445,
          5.762451739445877, 5.4460093013358035]]
    )
    df.columns = ['a', 'b', 'c', 'b_(-inf, 5]', 'b_(5, 10]', 'b_(10, 15]', 'b_(15, inf]',
                  'c_h_1_pred', 'factor_b', 'factor_b_(5, 10]', 'factor_b_(15, inf]',
                  'c_h_1_pred_b_11', 'c_h_1_pred_b_12', 'c_h_1_pred_b_13',
                  'c_h_1_pred_b_14', 'c_h_1_pred_b_15', 'c_h_1_pred_c_90']
    return df


@pytest.fixture()
def test_fd_1():
    return {
        "experiment_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": ["1970-01-01 00:00:07"],
            "validate_start": "1970-01-01 00:00:01",
            "validate_end": "1970-01-01 00:00:09",
            "forecast_start": "1970-01-01 00:00:05",
            "forecast_end": "1970-01-01 00:00:14",
            "frequency": "S",
            "confidence_intervals": [90],
            "bootstrap_sample": 5,
            "bin_features": {"b": [5, 10, 15]},
            "scenarios": {"b": {"mode": "constant", "constant_values": [0, 1, 2, 3, 4, 5]}},
            "time_horizons": [1],
            "data_path": "divina-test/dataset/test1"
        }
    }


@pytest.fixture()
def test_fds_quickstart():
    fds = {}
    for file in sorted(
            os.listdir(pathlib.Path(pathlib.Path(__file__).parent.parent.parent, 'docs_src/_static/experiment_definitions'))):
        with open(pathlib.Path(pathlib.Path(__file__).parent.parent.parent, 'docs_src/_static/experiment_definitions',
                               file)) as f:
            fds[file.split('.')[0]] = (json.load(f))
    return fds


@pytest.fixture()
def test_fd_2():
    return {
        "experiment_definition": {
            "time_index": "a",
            "target": "c",
            "time_validation_splits": ["1970-01-01 00:00:08"],
            "time_horizons": [1],
            "data_path": "divina-test/dataset/test1",
            "joins": [
                {
                    "data_path": "dataset/test2",
                    "join_on": ("a", "a"),
                    "as": "test2",
                }
            ],
        }
    }


@pytest.fixture()
def test_fd_3(test_bucket, test_fd_1):
    test_fd = test_fd_1
    test_fd["experiment_definition"].update({"data_path": "{}/dataset/test1".format(test_bucket)})
    return test_fd


@pytest.fixture()
def test_composite_dataset_1():
    df = pd.DataFrame(
        [[Timestamp("1970-01-01 00:00:01"), 8.0, 12.0, 2.0, 3.0],
         [Timestamp("1970-01-01 00:00:04"), 20.0, 24.0, np.NaN, 6.0],
         [Timestamp("1970-01-01 00:00:05"), 15.0, 18.0, np.NaN, np.NaN],
         [Timestamp("1970-01-01 00:00:06"), 5.0, 6.0, np.NaN, np.NaN],
         [Timestamp("1970-01-01 00:00:07"), 48.0, 54.0, 8.0, np.NaN],
         [Timestamp("1970-01-01 00:00:10"), 77.0, 84.0, np.NaN, np.NaN]]
    )
    df.columns = ["a", "b", "c", "e", "f"]
    return df


@pytest.fixture()
def test_df_1():
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
    return df


@pytest.fixture()
def test_df_2():
    df = pd.DataFrame(
        [[Timestamp("1970-01-01 00:00:01"), 2.0, 3.0], [Timestamp("1970-01-01 00:00:04"), np.NaN, 6.0],
         [Timestamp("1970-01-01 00:00:07"), 8.0, np.NaN], [np.NaN, 11.0, 12.0]]
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
