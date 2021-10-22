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


@pytest.fixture()
def test_bucket():
    return "s3://divina-test-2"


@pytest.fixture()
def random_state():
    return 11


@pytest.fixture()
def setup_teardown_test_bucket_contents(s3_fs, test_bucket):
    fsspec.filesystem('s3').invalidate_cache()
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


@pytest.fixture()
def dask_client(request):
    client = Client()
    request.addfinalizer(lambda: client.close())
    yield client
    client.shutdown()


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
    cluster.scale(5)
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
    for c in ['b']:
        edges = [-np.inf] + [5, 10, 15] + [np.inf]
        for v, v_1 in zip(edges, edges[1:]):
            train_df["{}_({}, {}]".format(c, v, v_1)] = 1
            train_df["{}_({}, {}]".format(c, v, v_1)] = train_df["{}_({}, {}]".format(c, v, v_1)].where(
                ((train_df[c] < v_1) & (train_df[c] >= v)), 0)
    model = LinearRegression()
    features = [c for c in train_df.columns if not c in ['c', 'a'] and train_df[c].nunique() != 1]
    model.fit(
        ddf.from_pandas(train_df[features], chunksize=10000).to_dask_array(
            lengths=True),
        ddf.from_pandas(train_df, chunksize=10000)["c"].to_dask_array(lengths=True),
    )
    return (model, {"params": {f: c for f, c in zip(features, model._coef)}})


@pytest.fixture()
def test_model_retail(test_df_retail_sales, test_df_retail_time, test_df_retail_stores, test_fd_retail):
    train_df = test_df_retail_sales.join(test_df_retail_time.set_index('date'), on='Date').join(
        test_df_retail_stores.set_index('Store'), on='Store')
    train_df['Sales'] = train_df['Sales'].shift(-2)
    train_df = train_df[train_df['Date'] < "2015-07-18"]
    train_df = pd.get_dummies(train_df, columns=test_fd_retail['forecast_definition']['encode_features'])
    features = [c for c in train_df if
                not c in test_fd_retail['forecast_definition']['drop_features'] + ['Date', 'Sales'] and train_df[
                    c].nunique() != 1]
    for c in train_df:
        if train_df[c].dtype == bool:
            train_df[c] = train_df[c].astype(int)
    model = LinearRegression()
    model.fit(
        ddf.from_pandas(train_df[features], chunksize=10000).to_dask_array(
            lengths=True),
        ddf.from_pandas(train_df, chunksize=10000)["Sales"].to_dask_array(lengths=True),
    )
    return (model, {"params": {f: c for f, c in zip(features, model._coef)}})


@pytest.fixture()
def test_params_1(test_model_1):
    return test_model_1[1]


@pytest.fixture()
def test_bootstrap_models(test_df_1, random_state, test_fd_1):
    train_df = test_df_1.groupby('a').agg('sum').reset_index()
    train_df['c'] = train_df['c'].shift(-1)
    train_df = train_df[train_df['a'] < "1970-01-01 00:00:07"]
    for c in ['b']:
        edges = [-np.inf] + [5, 10, 15] + [np.inf]
        for v, v_1 in zip(edges, edges[1:]):
            train_df["{}_({}, {}]".format(c, v, v_1)] = 1
            train_df["{}_({}, {}]".format(c, v, v_1)] = train_df["{}_({}, {}]".format(c, v, v_1)].where(
                ((train_df[c] < v_1) & (train_df[c] >= v)), 0)
    bootstrap_models = {}
    for rs in range(random_state, random_state + test_fd_1['forecast_definition']['bootstrap_sample']):
        model = LinearRegression()
        confidence_df = train_df.sample(frac=.8, random_state=rs)
        features = [c for c in confidence_df.columns if not c in ['c', 'a'] and confidence_df[c].nunique() != 1]
        model.fit(
            ddf.from_pandas(confidence_df[features], chunksize=10000)[features
            ].to_dask_array(lengths=True),
            ddf.from_pandas(confidence_df, chunksize=10000)["c"].to_dask_array(lengths=True),
        )
        bootstrap_models[rs] = (model, {"params": {f: c for f, c in zip(features, model._coef)}})
    return bootstrap_models


@pytest.fixture()
def test_bootstrap_models_retail(test_df_retail_sales, test_df_retail_stores, test_df_retail_time, random_state,
                                 test_fd_retail):
    train_df = test_df_retail_sales.join(test_df_retail_time.set_index('date'), on='Date').join(
        test_df_retail_stores.set_index('Store'), on='Store')
    train_df['Sales'] = train_df['Sales'].shift(-2)
    train_df = train_df[train_df['Date'] < "2015-07-18"]
    train_df = pd.get_dummies(train_df, columns=test_fd_retail['forecast_definition']['encode_features'])
    for c in train_df:
        if train_df[c].dtype == bool:
            train_df[c] = train_df[c].astype(int)
    bootstrap_models = {}
    for rs in range(random_state, random_state + test_fd_retail['forecast_definition']['bootstrap_sample']):
        model = LinearRegression()
        confidence_df = train_df.sample(frac=.8, random_state=rs)
        features = [c for c in confidence_df if
                    not c in test_fd_retail['forecast_definition']['drop_features'] + ['Sales', 'Date'] and
                    confidence_df[
                        c].nunique() != 1]
        model.fit(
            ddf.from_pandas(confidence_df[features], chunksize=10000).to_dask_array(lengths=True),
            ddf.from_pandas(confidence_df, chunksize=10000)["Sales"].to_dask_array(lengths=True),
        )
        bootstrap_models[rs] = (model, {"params": {f: c for f, c in zip(features, model._coef)}})
    return bootstrap_models


@pytest.fixture()
def test_params_2(test_model_1):
    return {"params": {c: test_model_1[1]['params'][c] + 1 for c in test_model_1[1]['params']}}


@pytest.fixture()
def test_metrics_1():
    return {'splits': {'1970-01-01 00:00:07': {'time_horizons': {'1': {'mae': 14.686136955152564}}}}}


@pytest.fixture()
def test_metrics_retail():
    return {'splits': {'2015-07-18': {'time_horizons': {'2': {'mae': 4162.4048830079555}}}}}


@pytest.fixture()
def test_val_predictions_1():
    df = pd.DataFrame(
        [[Timestamp('1970-01-01 00:00:01'), 34.17995524296469], [Timestamp('1970-01-01 00:00:04'), 7.684012803524567],
         [Timestamp('1970-01-01 00:00:05'), 11.577975376826522], [Timestamp('1970-01-01 00:00:06'), 36.51633278694585],
         [Timestamp('1970-01-01 00:00:07'), -14.12217760696636]]
    )
    df.columns = ["a", "c_h_1_pred"]
    return df


@pytest.fixture()
def test_val_predictions_retail():
    df = pd.DataFrame(
        [[Timestamp('2015-07-19 00:00:00'), 2.9979032274495694], [Timestamp('2015-07-21 00:00:00'), 7.146434722515551],
         [Timestamp('2015-07-23 00:00:00'), 10.897480974676], [Timestamp('2015-07-25 00:00:00'), 14.402110557140986],
         [Timestamp('2015-07-27 00:00:00'), 14.714350657646719], [Timestamp('2015-07-28 00:00:00'), 16.13033999428808],
         [Timestamp('2015-07-30 00:00:00'), 19.561710566838883], [Timestamp('2015-07-20 00:00:00'), 5.763744935833529],
         [Timestamp('2015-07-22 00:00:00'), 8.868779918779182], [Timestamp('2015-07-24 00:00:00'), 12.366749591255939],
         [Timestamp('2015-07-26 00:00:00'), 9.383130614867696], [Timestamp('2015-07-29 00:00:00'), 17.646227980803815],
         [Timestamp('2015-07-31 00:00:00'), 21.383954412987805], [Timestamp('2015-07-18 00:00:00'), 8.096802089628909]])
    df.columns = ['Date', 'Sales_h_2_pred']
    return df


@pytest.fixture()
def test_forecast_1():
    df = pd.DataFrame(
        [[Timestamp('1970-01-01 00:00:05'), 11.577975376826522, 21.69483124057578],
         [Timestamp('1970-01-01 00:00:06'), 36.51633278694585, 47.70685132054264],
         [Timestamp('1970-01-01 00:00:07'), -14.12217760696636, 35.141751426272975],
         [Timestamp('1970-01-01 00:00:10'), 31.835024241839548, 55.693340543877525],
         [Timestamp('1970-01-01 00:00:10'), 31.835024241839548, 55.693340543877525],
         [Timestamp('1970-01-01 00:00:11'), 31.835024241839548, 55.693340543877525],
         [Timestamp('1970-01-01 00:00:12'), 31.835024241839548, 55.693340543877525],
         [Timestamp('1970-01-01 00:00:13'), 31.835024241839548, 55.693340543877525],
         [Timestamp('1970-01-01 00:00:14'), 31.835024241839548, 55.693340543877525],
         [Timestamp('1970-01-01 00:00:10'), 31.056231727179156, 52.22393002379304],
         [Timestamp('1970-01-01 00:00:10'), 31.056231727179156, 52.22393002379304],
         [Timestamp('1970-01-01 00:00:11'), 31.056231727179156, 52.22393002379304],
         [Timestamp('1970-01-01 00:00:12'), 31.056231727179156, 52.22393002379304],
         [Timestamp('1970-01-01 00:00:13'), 31.056231727179156, 52.22393002379304],
         [Timestamp('1970-01-01 00:00:14'), 31.056231727179156, 52.22393002379304],
         [Timestamp('1970-01-01 00:00:10'), 30.277439212518768, 48.75451950370855],
         [Timestamp('1970-01-01 00:00:10'), 30.277439212518768, 48.75451950370855],
         [Timestamp('1970-01-01 00:00:11'), 30.277439212518768, 48.75451950370855],
         [Timestamp('1970-01-01 00:00:12'), 30.277439212518768, 48.75451950370855],
         [Timestamp('1970-01-01 00:00:13'), 30.277439212518768, 48.75451950370855],
         [Timestamp('1970-01-01 00:00:14'), 30.277439212518768, 48.75451950370855],
         [Timestamp('1970-01-01 00:00:10'), 29.498646697858376, 45.285108983624056],
         [Timestamp('1970-01-01 00:00:10'), 29.498646697858376, 45.285108983624056],
         [Timestamp('1970-01-01 00:00:11'), 29.498646697858376, 45.285108983624056],
         [Timestamp('1970-01-01 00:00:12'), 29.498646697858376, 45.285108983624056],
         [Timestamp('1970-01-01 00:00:13'), 29.498646697858376, 45.285108983624056],
         [Timestamp('1970-01-01 00:00:14'), 29.498646697858376, 45.285108983624056],
         [Timestamp('1970-01-01 00:00:10'), 28.719854183197988, 41.81569846353957],
         [Timestamp('1970-01-01 00:00:10'), 28.719854183197988, 41.81569846353957],
         [Timestamp('1970-01-01 00:00:11'), 28.719854183197988, 41.81569846353957],
         [Timestamp('1970-01-01 00:00:12'), 28.719854183197988, 41.81569846353957],
         [Timestamp('1970-01-01 00:00:13'), 28.719854183197988, 41.81569846353957],
         [Timestamp('1970-01-01 00:00:14'), 28.719854183197988, 41.81569846353957],
         [Timestamp('1970-01-01 00:00:10'), 36.51633278694585, 47.70685132054264],
         [Timestamp('1970-01-01 00:00:10'), 36.51633278694585, 47.70685132054264],
         [Timestamp('1970-01-01 00:00:11'), 36.51633278694585, 47.70685132054264],
         [Timestamp('1970-01-01 00:00:12'), 36.51633278694585, 47.70685132054264],
         [Timestamp('1970-01-01 00:00:13'), 36.51633278694585, 47.70685132054264],
         [Timestamp('1970-01-01 00:00:14'), 36.51633278694585, 47.70685132054264]]
    )
    df.index = list(df.index + 1)
    df.index.name = 'forecast_index'
    df.columns = ["a", "c_h_1_pred", "c_h_1_pred_c_90"]
    return df


@pytest.fixture()
def test_forecast_retail():
    df = pd.DataFrame([[Timestamp('2015-07-21 00:00:00'), 7.146434722515551, 9.542689375084592, 5.209320027148351],
                       [Timestamp('2015-07-23 00:00:00'), 10.897480974676, 13.308263108221581, 9.774275952666358],
                       [Timestamp('2015-07-25 00:00:00'), 14.402110557140986, 16.843613759767322, 13.415718525709963],
                       [Timestamp('2015-07-27 00:00:00'), 14.714350657646719, 17.885705157732446, 13.933398127832334],
                       [Timestamp('2015-07-28 00:00:00'), 16.13033999428808, 19.339157088372303, 15.34090421541623],
                       [Timestamp('2015-07-30 00:00:00'), 19.561710566838883, 22.80606303998502, 18.698973878235847],
                       [Timestamp('2015-07-22 00:00:00'), 8.868779918779182, 11.282364596339903, 6.917722122147097],
                       [Timestamp('2015-07-24 00:00:00'), 12.366749591255939, 14.81149300243851, 10.53420487484982],
                       [Timestamp('2015-07-26 00:00:00'), 9.383130614867696, 12.352378437008156, 2.7939693740190705],
                       [Timestamp('2015-07-29 00:00:00'), 17.646227980803815, 20.885942700730084, 16.575976495958457],
                       [Timestamp('2015-07-31 00:00:00'), 21.383954412987805, 24.639071942969167, 20.474552263000078]])
    df.index = list(df.index + 1)
    df.index.name = 'forecast_index'
    df.columns = ['Date', 'Sales_h_2_pred', 'Sales_h_2_pred_c_90', 'Sales_h_2_pred_c_10']
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
            "forecast_freq": "S",
            "confidence_intervals": [90],
            "bootstrap_sample": 5,
            "bin_features": {"b": [5, 10, 15]},
            "scenarios": {"b": {"values": [[0, 5]], "start": "1970-01-01 00:00:09", "end": "1970-01-01 00:00:14"}},
            "time_horizons": [1],
            "dataset_directory": "divina-test/dataset/test1",
            "model": "LinearRegression",
        }
    }


@pytest.fixture()
def test_fd_retail():
    return {
        "forecast_definition": {
            "time_index": "Date",
            "target": "Sales",
            "drop_features": ['date', 'holiday_type', 'Promo2SinceWeek', 'Promo2SinceYear'],
            "time_validation_splits": ["2015-07-18"],
            "forecast_start": "2015-07-21",
            "forecast_end": "2015-07-31",
            "forecast_freq": 'D',
            "bootstrap_sample": 5,
            "time_horizons": [2],
            "encode_features": ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval', 'Store', 'weekday'],
            "dataset_directory": "dataset/retail/sales2",
            "confidence_intervals": [90, 10],
            "interaction_terms": {'weekday_5': '*'},
            "joins": [
                {
                    "dataset_directory": "dataset/time",
                    "join_on": ("Date", "date"),
                    "as": "time"
                },
                {
                    "dataset_directory": "dataset/retail/store",
                    "join_on": ("Store", "Store"),
                    "as": "store"
                }
            ],
            "model": "LinearRegression",
        }
    }


@pytest.fixture()
def test_fd_retail_2(test_bucket, test_fd_retail):
    test_fd = test_fd_retail
    test_fd["forecast_definition"].update(
        {"dataset_directory": "{}/{}".format(test_bucket, test_fd['forecast_definition']['dataset_directory'])})
    for join in test_fd["forecast_definition"]["joins"]:
        join.update({"dataset_directory": "{}/{}".format(test_bucket, join['dataset_directory'])})
    return test_fd


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
def test_df_retail_sales():
    df = pd.DataFrame([[1, 5, '2015-07-31', 5263, 555, 1, 1, 'z', 1],
                       [1, 4, '2015-07-30', 5020, 546, 1, 1, 'z', 1],
                       [1, 3, '2015-07-29', 4782, 523, 1, 1, 'z', 1],
                       [1, 2, '2015-07-28', 5011, 560, 1, 1, 'z', 1],
                       [1, 1, '2015-07-27', 6102, 612, 1, 1, 'z', 1],
                       [1, 7, '2015-07-26', 0, 0, 0, 0, 'z', 0],
                       [1, 6, '2015-07-25', 4364, 500, 1, 0, 'z', 0],
                       [1, 5, '2015-07-24', 3706, 459, 1, 0, 'z', 0],
                       [1, 4, '2015-07-23', 3769, 503, 1, 0, 'z', 0],
                       [1, 3, '2015-07-22', 3464, 463, 1, 0, 'z', 0],
                       [1, 2, '2015-07-21', 3558, 469, 1, 0, 'z', 0],
                       [1, 1, '2015-07-20', 4395, 526, 1, 0, 'z', 0],
                       [1, 7, '2015-07-19', 0, 0, 0, 0, 'z', 0],
                       [1, 6, '2015-07-18', 4406, 512, 1, 0, 'z', 0],
                       [1, 5, '2015-07-17', 4852, 519, 1, 1, 'z', 0],
                       [1, 4, '2015-07-16', 4427, 517, 1, 1, 'z', 0],
                       [1, 3, '2015-07-15', 4767, 550, 1, 1, 'z', 0],
                       [1, 2, '2015-07-14', 5042, 544, 1, 1, 'z', 0],
                       [1, 1, '2015-07-13', 5054, 553, 1, 1, 'z', 0],
                       [1, 7, '2015-07-12', 0, 0, 0, 0, 'z', 0],
                       [1, 6, '2015-07-11', 3530, 441, 1, 0, 'z', 0],
                       [1, 5, '2015-07-10', 3808, 449, 1, 0, 'z', 0],
                       [1, 4, '2015-07-09', 3897, 480, 1, 0, 'z', 0],
                       [1, 3, '2015-07-08', 3797, 485, 1, 0, 'z', 0],
                       [1, 2, '2015-07-07', 3650, 485, 1, 0, 'z', 0]])
    df.columns = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
                  'StateHoliday', 'SchoolHoliday']
    return df


@pytest.fixture()
def test_df_retail_stores():
    df = pd.DataFrame([[1.0, 'c', 'a', 1270.0, 9.0, 2008, 0.0, np.NaN, np.NaN, None]])
    df.columns = ['Store', 'StoreType', 'Assortment', 'CompetitionDistance',
                  'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
                  'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
    return df


@pytest.fixture()
def test_df_retail_time():
    df = pd.DataFrame([['2015-07-07', 7, 7, 2015, 1, False, None, 28, 78714],
                       ['2015-07-08', 7, 8, 2015, 2, False, None, 28, 78715],
                       ['2015-07-09', 7, 9, 2015, 3, False, None, 28, 78716],
                       ['2015-07-10', 7, 10, 2015, 4, False, None, 28, 78717],
                       ['2015-07-11', 7, 11, 2015, 5, False, None, 28, 78718],
                       ['2015-07-12', 7, 12, 2015, 6, False, None, 28, 78719],
                       ['2015-07-13', 7, 13, 2015, 0, False, None, 29, 78720],
                       ['2015-07-14', 7, 14, 2015, 1, False, None, 29, 78721],
                       ['2015-07-15', 7, 15, 2015, 2, False, None, 29, 78722],
                       ['2015-07-16', 7, 16, 2015, 3, False, None, 29, 78723],
                       ['2015-07-17', 7, 17, 2015, 4, False, None, 29, 78724],
                       ['2015-07-18', 7, 18, 2015, 5, False, None, 29, 78725],
                       ['2015-07-19', 7, 19, 2015, 6, False, None, 29, 78726],
                       ['2015-07-20', 7, 20, 2015, 0, False, None, 30, 78727],
                       ['2015-07-21', 7, 21, 2015, 1, False, None, 30, 78728],
                       ['2015-07-22', 7, 22, 2015, 2, False, None, 30, 78729],
                       ['2015-07-23', 7, 23, 2015, 3, False, None, 30, 78730],
                       ['2015-07-24', 7, 24, 2015, 4, False, None, 30, 78731],
                       ['2015-07-25', 7, 25, 2015, 5, False, None, 30, 78732],
                       ['2015-07-26', 7, 26, 2015, 6, False, None, 30, 78733],
                       ['2015-07-27', 7, 27, 2015, 0, False, None, 31, 78734],
                       ['2015-07-28', 7, 28, 2015, 1, False, None, 31, 78735],
                       ['2015-07-29', 7, 29, 2015, 2, False, None, 31, 78736],
                       ['2015-07-30', 7, 30, 2015, 3, False, None, 31, 78737],
                       ['2015-07-31', 7, 31, 2015, 4, False, None, 31, 78738]])
    df.columns = ['date', 'month', 'day', 'year', 'weekday', 'holiday', 'holiday_type',
                  'week_of_year', 't']
    return df
