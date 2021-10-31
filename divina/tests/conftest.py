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
def test_model_1(test_df_1, random_state, test_fd_1):
    params = [31.835024241839548, -0.7787925146603903, 8.57527111840826]
    intercept = -8.575161145107172
    features = ["b", "b_(5, 10]", "b_(15, inf]"]

    model = LinearRegression()
    model.fit(
        ddf.from_pandas(pd.DataFrame([np.array(params) + c for c in range(0, 2)]), npartitions=1).to_dask_array(
            lengths=True),
        ddf.from_pandas(pd.Series([intercept, intercept]), npartitions=1).to_dask_array(lengths=True))
    model.coef_ = np.array(params)
    model.intercept_ = intercept
    model._coef = np.array(params + [intercept])
    return (model, {"params": {f: c for f, c in zip(features, params)}})


@pytest.fixture()
def test_model_retail(test_df_1, random_state, test_fd_1):
    params = [7.037158170969071, 0.4932340350688223, 0.27259342357812283, 0.18940264900594353, -3.6834622221000717,
              0.4984087190231185, 0.5368146044322767, 0.49909604923189516, 0.487186048658764, 0.6179858352023053,
              0.5021278348261174, -3.385064917603151, 0.5330286617997554, 0.41455435987080824, 0.18775054632565508,
              0.15306128463281526, 0.6284985207120164, 0.6795648992676631, 0.40046449375304205, -3.359386388458164,
              0.4558462123976466, 0.32753112486200764, 0.4047017669158538, 0.7156131294903266, 0.4747486484045316,
              0.45976607692456334, 0.5064120143313671, 0.6447829633588602, -3.3850649176031515, 0.19104786904854365,
              0.18940264900594353, 0.6928306163987392, 0.6569387471664435, 0.34607755993840705, 0.41276666500878173]
    intercept = 0.6542184768053281
    features = ['Promo', 'SchoolHoliday', 'LastDayOfMonth', 'Weekday_6.0', 'Weekday_3.0', 'Weekday_0.0', 'Weekday_1.0', 'Weekday_2.0', 'Weekday_5.0', 'Weekday_4.0', 'DayOfMonth_19.0', 'DayOfMonth_23.0', 'DayOfMonth_27.0', 'DayOfMonth_28.0', 'DayOfMonth_29.0', 'DayOfMonth_7.0', 'DayOfMonth_8.0', 'DayOfMonth_11.0', 'DayOfMonth_12.0', 'DayOfMonth_14.0', 'DayOfMonth_16.0', 'DayOfMonth_17.0', 'DayOfMonth_20.0', 'DayOfMonth_21.0', 'DayOfMonth_22.0', 'DayOfMonth_24.0', 'DayOfMonth_25.0', 'DayOfMonth_26.0', 'DayOfMonth_30.0', 'DayOfMonth_31.0', 'DayOfMonth_9.0', 'DayOfMonth_10.0', 'DayOfMonth_13.0', 'DayOfMonth_15.0', 'DayOfMonth_18.0']

    model = LinearRegression()
    model.fit(
        ddf.from_pandas(pd.DataFrame([np.array(params) + c for c in range(0, 2)]), npartitions=1).to_dask_array(
            lengths=True),
        ddf.from_pandas(pd.Series([intercept, intercept]), npartitions=1).to_dask_array(lengths=True))
    model.coef_ = np.array(params)
    model.intercept_ = intercept
    model._coef = np.array(params + [intercept])
    return (model, {"params": {f: c for f, c in zip(features, params)}})


@pytest.fixture()
def test_params_1(test_model_1):
    return test_model_1[1]


@pytest.fixture()
def test_bootstrap_models(test_df_1, random_state, test_fd_1):
    params = [[55.69334054387751, -3.4694105200844874, 1.5010004670187311],
              [27.650949293638853, 0.0764031931120413, 19.67388606134358],
              [27.65094929363883, 0.07640319311204169, 19.673886061343605],
              [46.177552894648386, -1.6952607465242302, -0.9461935879036099],
              [4.888905149914471, 0.8196202877424658, 9.088895999249898]]
    intercepts = [-1.5010264833024487, -19.674023685557096, -19.67402368555708, 0.946189543790848, -9.088927535279836]
    features = [["b", "b_(5, 10]", "b_(15, inf]"], ["b", "b_(5, 10]", "b_(15, inf]"], ["b", "b_(5, 10]", "b_(15, inf]"],
                ["b", "b_(5, 10]", "b_(15, inf]"], ["b", "b_(5, 10]", "b_(15, inf]"]]
    seeds = range(random_state, random_state + test_fd_1["forecast_definition"]["bootstrap_sample"])
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
        bootstrap_models[seed] = (model, {"params": {_f: c for _f, c in zip(f, p)}})
    return bootstrap_models


@pytest.fixture()
def test_bootstrap_models_retail(test_df_1, random_state, test_fd_1):
    params = [[6.643932953270661, 0.6260364498868285, 0.31918623945495733, 0.23423245984677996, -3.547744343528209,
               0.5963643382718036, 0.5461500634517222, 0.599524608261714, 0.6256869892244343, 0.7922640044073792,
               0.564301267830736, -3.196111422680582, 0.4778005895412411, 0.16215693805253273, 0.15712633420359842,
               0.8892764359349725, 0.865254690614509, 0.5854459099996437, -3.1306691705249476, 0.5578175859834436,
               0.44678390027838205, 0.6447735626026266, 0.8336877386889633, 0.7659003606714241, -3.196111422680582,
               0.22863089991643462, 0.23423245984677996, 0.9013229016856845, 0.5570103755595892, 0.4404237454134457],
              [6.688561227702059, 0.5517670506412442, 0.35321326079971904, 0.1860355789654757, -3.5716807262469623,
               0.6312114345896419, 0.5749344154496477, 0.6004142467183095, 0.5750982269355176, 0.7683271180577859,
               0.630062792820489, -3.226265416798491, 0.6708709221395525, 0.5749344154496477, 0.18613624126348308,
               0.22240616547787925, 0.8591656256802459, 0.8672504393401864, 0.5814917292774523, -3.134619142753566,
               0.5663293999281276, 0.3733411776795603, 0.4539281460296921, 0.656652627308842, 0.7357478037278692,
               -3.2262654167984905, 0.2247350368500566, 0.1860355789654757, 0.8572276608371586, 0.8256893360850589],
              [7.218826725057033, 0.44606980581521916, 0.1934039445686336, -3.7260290898011217, 0.3824849821099785,
               0.47028028452181536, 0.4132329845511205, 0.4011327981897966, 0.6208722632334958, 0.45547992921194,
               -3.533626570413229, 0.5504963908320178, 0.15133052926678806, 0.11697390204336387, 0.43925745124993065,
               0.49028930152094424, -3.5262178280455587, 0.40510108956393864, 0.22534135177343922, 0.39544415418566453,
               0.5446576863156511, 0.3962542838852929, 0.38151727039285593, 0.3183278938569467, 0.5834979613429083,
               0.25083289345462256, 0.5088323933964385, 0.3463998876269044, 0.3624043575286703],
              [7.157878693809477, 0.4166532356186096, 0.2633685685801884, 0.164178471705681, -3.7097936844543056,
               0.4402550155139545, 0.5157339278355035, 0.4163708721720424, 0.41258713363346466, 0.4897833764933656,
               0.42595776984871475, 0.4427894415100444, 0.3970143113481576, 0.1170006963015829, 0.1423792862374243,
               0.5833465393467204, 0.5878464170288371, 0.44149331952488113, -3.4315340053970713, 0.4172853054902338,
               0.319890581030604, 0.5801682640814287, 0.33113254089743094, 0.38586624762663324, 0.48651979020392255,
               -3.59754626351183, 0.21905209471883844, 0.164178471705681, 0.5199010738801335, 0.5648785042573997],
              [6.739881079307763, 0.5124317583959213, 0.32874966824916374, 0.21786124859964803, -3.5971338819678467,
               0.6571444169347329, 0.6393290714677631, 0.6054418544693634, 0.5723254184047881, 0.7428770450458657,
               0.5383230421051751, -3.2344106484131645, 0.5529381588508739, 0.45095914279593674, 0.2833780811611728,
               0.15515672961777646, 0.681587260724348, 0.8524013200132534, 0.5294578008132041, -3.1866509458577976,
               0.39477883614026943, 0.8416858340931691, 0.6601604448144781, 0.7276077642038079, -3.234410648413166,
               0.21786124859964803, 0.8161893541158314, 0.8021157665835167, 0.42344123438543696, 0.5286762507902876]]
    intercepts = [0.7752578052698612, 0.7451052306199654, 0.5928917827900219, 0.32841762794188634, 0.736965206914541]
    features = [['Promo', 'SchoolHoliday', 'LastDayOfMonth', 'Weekday_6.0', 'Weekday_3.0', 'Weekday_0.0', 'Weekday_1.0',
                 'Weekday_2.0', 'Weekday_5.0', 'Weekday_4.0', 'DayOfMonth_19.0', 'DayOfMonth_27.0', 'DayOfMonth_28.0',
                 'DayOfMonth_29.0', 'DayOfMonth_7.0', 'DayOfMonth_8.0', 'DayOfMonth_11.0', 'DayOfMonth_12.0',
                 'DayOfMonth_14.0', 'DayOfMonth_17.0', 'DayOfMonth_22.0', 'DayOfMonth_24.0', 'DayOfMonth_25.0',
                 'DayOfMonth_26.0', 'DayOfMonth_30.0', 'DayOfMonth_31.0', 'DayOfMonth_9.0', 'DayOfMonth_13.0',
                 'DayOfMonth_15.0', 'DayOfMonth_18.0'],
                ['Promo', 'SchoolHoliday', 'LastDayOfMonth', 'Weekday_6.0', 'Weekday_3.0', 'Weekday_0.0', 'Weekday_1.0',
                 'Weekday_2.0', 'Weekday_5.0', 'Weekday_4.0', 'DayOfMonth_19.0', 'DayOfMonth_23.0', 'DayOfMonth_27.0',
                 'DayOfMonth_28.0', 'DayOfMonth_29.0', 'DayOfMonth_7.0', 'DayOfMonth_8.0', 'DayOfMonth_11.0',
                 'DayOfMonth_12.0', 'DayOfMonth_14.0', 'DayOfMonth_16.0', 'DayOfMonth_17.0', 'DayOfMonth_24.0',
                 'DayOfMonth_25.0', 'DayOfMonth_26.0', 'DayOfMonth_30.0', 'DayOfMonth_31.0', 'DayOfMonth_9.0',
                 'DayOfMonth_10.0', 'DayOfMonth_18.0'],
                ['Promo', 'SchoolHoliday', 'Weekday_6.0', 'Weekday_3.0', 'Weekday_0.0', 'Weekday_1.0', 'Weekday_2.0',
                 'Weekday_5.0', 'Weekday_4.0', 'DayOfMonth_19.0', 'DayOfMonth_23.0', 'DayOfMonth_28.0',
                 'DayOfMonth_29.0', 'DayOfMonth_7.0', 'DayOfMonth_8.0', 'DayOfMonth_12.0', 'DayOfMonth_14.0',
                 'DayOfMonth_16.0', 'DayOfMonth_17.0', 'DayOfMonth_20.0', 'DayOfMonth_21.0', 'DayOfMonth_22.0',
                 'DayOfMonth_24.0', 'DayOfMonth_25.0', 'DayOfMonth_30.0', 'DayOfMonth_10.0', 'DayOfMonth_13.0',
                 'DayOfMonth_15.0', 'DayOfMonth_18.0'],
                ['Promo', 'SchoolHoliday', 'LastDayOfMonth', 'Weekday_6.0', 'Weekday_3.0', 'Weekday_0.0', 'Weekday_1.0',
                 'Weekday_2.0', 'Weekday_5.0', 'Weekday_4.0', 'DayOfMonth_23.0', 'DayOfMonth_27.0', 'DayOfMonth_28.0',
                 'DayOfMonth_29.0', 'DayOfMonth_7.0', 'DayOfMonth_8.0', 'DayOfMonth_11.0', 'DayOfMonth_12.0',
                 'DayOfMonth_14.0', 'DayOfMonth_17.0', 'DayOfMonth_20.0', 'DayOfMonth_22.0', 'DayOfMonth_24.0',
                 'DayOfMonth_25.0', 'DayOfMonth_26.0', 'DayOfMonth_30.0', 'DayOfMonth_31.0', 'DayOfMonth_9.0',
                 'DayOfMonth_10.0', 'DayOfMonth_15.0'],
                ['Promo', 'SchoolHoliday', 'LastDayOfMonth', 'Weekday_6.0', 'Weekday_3.0', 'Weekday_0.0', 'Weekday_1.0',
                 'Weekday_2.0', 'Weekday_5.0', 'Weekday_4.0', 'DayOfMonth_19.0', 'DayOfMonth_23.0', 'DayOfMonth_27.0',
                 'DayOfMonth_28.0', 'DayOfMonth_29.0', 'DayOfMonth_7.0', 'DayOfMonth_8.0', 'DayOfMonth_11.0',
                 'DayOfMonth_12.0', 'DayOfMonth_16.0', 'DayOfMonth_20.0', 'DayOfMonth_21.0', 'DayOfMonth_25.0',
                 'DayOfMonth_26.0', 'DayOfMonth_31.0', 'DayOfMonth_9.0', 'DayOfMonth_10.0', 'DayOfMonth_13.0',
                 'DayOfMonth_15.0', 'DayOfMonth_18.0']]
    seeds = range(random_state, random_state + test_fd_1["forecast_definition"]["bootstrap_sample"])
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
        bootstrap_models[seed] = (model, {"params": {_f: c for _f, c in zip(f, p)}})
    return bootstrap_models


@pytest.fixture()
def test_params_2(test_model_1):
    return {"params": {c: test_model_1[1]["params"][c] + 1 for c in test_model_1[1]["params"]}}


@pytest.fixture()
def test_metrics_1():
    return {"splits": {"1970-01-01 00:00:07": {"time_horizons": {"1": {"mae": 14.686136955152564}}}}}


@pytest.fixture()
def test_metrics_retail():
    return {"splits": {"2015-07-18": {"time_horizons": {"2": {"mae": 4162.277118203362}}}}}


@pytest.fixture()
def test_val_predictions_1():
    df = pd.DataFrame(
        [[Timestamp("1970-01-01 00:00:01"), 34.17995524296469], [Timestamp("1970-01-01 00:00:04"), 7.684012803524567],
         [Timestamp("1970-01-01 00:00:05"), 11.577975376826522], [Timestamp("1970-01-01 00:00:06"), 36.51633278694585],
         [Timestamp("1970-01-01 00:00:07"), -14.12217760696636]]
    )
    df.columns = ["a", "c_h_1_pred"]
    return df


@pytest.fixture()
def test_val_predictions_retail():
    df = pd.DataFrame(
        [[Timestamp("2015-07-19 00:00:00"), 1.8317264098620485], [Timestamp("2015-07-21 00:00:00"), 7.938372961307323],
         [Timestamp("2015-07-23 00:00:00"), 10.554629030437354], [Timestamp("2015-07-25 00:00:00"), 13.803328831072312],
         [Timestamp("2015-07-27 00:00:00"), 16.952544389150717], [Timestamp("2015-07-28 00:00:00"), 16.92227823307985],
         [Timestamp("2015-07-30 00:00:00"), 19.218858622600237], [Timestamp("2015-07-20 00:00:00"), 8.001938667333889],
         [Timestamp("2015-07-22 00:00:00"), 9.093323066059384], [Timestamp("2015-07-24 00:00:00"), 11.456502555503626],
         [Timestamp("2015-07-26 00:00:00"), 8.216953797283812], [Timestamp("2015-07-29 00:00:00"), 17.870771128084016],
         [Timestamp("2015-07-31 00:00:00"), 20.473707377235492], [Timestamp("2015-07-18 00:00:00"), 7.498020363556597]])
    df.columns = ["Date", "Sales_h_2_pred"]
    return df


@pytest.fixture()
def test_forecast_1():
    df = pd.DataFrame(
        [[Timestamp("1970-01-01 00:00:05"), 11.577975376826522, 21.69483124057578],
         [Timestamp("1970-01-01 00:00:06"), 36.51633278694585, 47.70685132054264],
         [Timestamp("1970-01-01 00:00:07"), -14.12217760696636, 35.141751426272975],
         [Timestamp("1970-01-01 00:00:10"), 31.835024241839548, 55.693340543877525],
         [Timestamp("1970-01-01 00:00:10"), 31.835024241839548, 55.693340543877525],
         [Timestamp("1970-01-01 00:00:11"), 31.835024241839548, 55.693340543877525],
         [Timestamp("1970-01-01 00:00:12"), 31.835024241839548, 55.693340543877525],
         [Timestamp("1970-01-01 00:00:13"), 31.835024241839548, 55.693340543877525],
         [Timestamp("1970-01-01 00:00:14"), 31.835024241839548, 55.693340543877525],
         [Timestamp("1970-01-01 00:00:10"), 31.056231727179156, 52.22393002379304],
         [Timestamp("1970-01-01 00:00:10"), 31.056231727179156, 52.22393002379304],
         [Timestamp("1970-01-01 00:00:11"), 31.056231727179156, 52.22393002379304],
         [Timestamp("1970-01-01 00:00:12"), 31.056231727179156, 52.22393002379304],
         [Timestamp("1970-01-01 00:00:13"), 31.056231727179156, 52.22393002379304],
         [Timestamp("1970-01-01 00:00:14"), 31.056231727179156, 52.22393002379304],
         [Timestamp("1970-01-01 00:00:10"), 30.277439212518768, 48.75451950370855],
         [Timestamp("1970-01-01 00:00:10"), 30.277439212518768, 48.75451950370855],
         [Timestamp("1970-01-01 00:00:11"), 30.277439212518768, 48.75451950370855],
         [Timestamp("1970-01-01 00:00:12"), 30.277439212518768, 48.75451950370855],
         [Timestamp("1970-01-01 00:00:13"), 30.277439212518768, 48.75451950370855],
         [Timestamp("1970-01-01 00:00:14"), 30.277439212518768, 48.75451950370855],
         [Timestamp("1970-01-01 00:00:10"), 29.498646697858376, 45.285108983624056],
         [Timestamp("1970-01-01 00:00:10"), 29.498646697858376, 45.285108983624056],
         [Timestamp("1970-01-01 00:00:11"), 29.498646697858376, 45.285108983624056],
         [Timestamp("1970-01-01 00:00:12"), 29.498646697858376, 45.285108983624056],
         [Timestamp("1970-01-01 00:00:13"), 29.498646697858376, 45.285108983624056],
         [Timestamp("1970-01-01 00:00:14"), 29.498646697858376, 45.285108983624056],
         [Timestamp("1970-01-01 00:00:10"), 28.719854183197988, 41.81569846353957],
         [Timestamp("1970-01-01 00:00:10"), 28.719854183197988, 41.81569846353957],
         [Timestamp("1970-01-01 00:00:11"), 28.719854183197988, 41.81569846353957],
         [Timestamp("1970-01-01 00:00:12"), 28.719854183197988, 41.81569846353957],
         [Timestamp("1970-01-01 00:00:13"), 28.719854183197988, 41.81569846353957],
         [Timestamp("1970-01-01 00:00:14"), 28.719854183197988, 41.81569846353957],
         [Timestamp("1970-01-01 00:00:10"), 36.51633278694585, 47.70685132054264],
         [Timestamp("1970-01-01 00:00:10"), 36.51633278694585, 47.70685132054264],
         [Timestamp("1970-01-01 00:00:11"), 36.51633278694585, 47.70685132054264],
         [Timestamp("1970-01-01 00:00:12"), 36.51633278694585, 47.70685132054264],
         [Timestamp("1970-01-01 00:00:13"), 36.51633278694585, 47.70685132054264],
         [Timestamp("1970-01-01 00:00:14"), 36.51633278694585, 47.70685132054264]]
    )
    df.index = list(df.index + 1)
    df.index.name = "forecast_index"
    df.columns = ["a", "c_h_1_pred", "c_h_1_pred_c_90"]
    return df


@pytest.fixture()
def test_forecast_retail():
    df = pd.DataFrame([[Timestamp("2015-07-21 00:00:00"), 7.938372961307323, 10.05660735992933, 5.864523315639474],
                       [Timestamp("2015-07-23 00:00:00"), 10.554629030437354, 12.762422159223206, 9.937192645555214],
                       [Timestamp("2015-07-25 00:00:00"), 13.803328831072312, 16.008575798354855, 9.652716573065845],
                       [Timestamp("2015-07-27 00:00:00"), 16.95254438914708, 19.700064530923555, 15.878979830766184],
                       [Timestamp("2015-07-28 00:00:00"), 16.92227823307985, 19.853075073217042, 15.955625570939446],
                       [Timestamp("2015-07-30 00:00:00"), 19.218858622600237, 22.26022209099392, 18.96366030318677],
                       [Timestamp("2015-07-22 00:00:00"), 9.093323066059384, 12.0369650356879, 6.941501152124715],
                       [Timestamp("2015-07-24 00:00:00"), 11.456502555503626, 14.5063345079476, 9.295135387800357],
                       [Timestamp("2015-07-26 00:00:00"), 8.216953797283812, 10.987461008674131, -2.094508001193674],
                       [Timestamp("2015-07-29 00:00:00"), 17.870771128084016, 21.64054314007808, 16.599755525935592],
                       [Timestamp("2015-07-31 00:00:00"), 20.473707377235492, 24.33391344847462, 19.408378340531314]])
    df.index = list(df.index + 1)
    df.index.name = "forecast_index"
    df.columns = ["Date", "Sales_h_2_pred", "Sales_h_2_pred_c_90", "Sales_h_2_pred_c_10"]
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
            "scenarios": [
                {"feature": "b", "values": [[0, 5]], "start": "1970-01-01 00:00:09", "end": "1970-01-01 00:00:14"}],
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
            "include_features": ["Store", "Promo", "StateHoliday", "SchoolHoliday", "Weekday",
                                 "LastDayOfMonth", "DayOfMonth"],
            "time_validation_splits": ["2015-07-18"],
            "forecast_end": "2015-08-30",
            "bootstrap_sample": 5,
            "signal_dimensions": ["Store"],
            "time_horizons": [2],
            "forecast_freq": "D",
            "encode_features": ["StateHoliday", "Weekday", "DayOfMonth"],
            "scenarios": [{"feature": "Promo", "values": [0, 1], "start": "2015-08-01", "end": "2016-01-01"},
                          {"feature": "StateHoliday", "values": ["z"], "start": "2015-08-01", "end": "2016-01-01"},
                          {"feature": "SchoolHoliday", "values": [0], "start": "2015-08-01", "end": "2016-01-01"}],
            "dataset_directory": "dataset/retail/sales2",
            "link_function": "log",
            "confidence_intervals": [90, 10],
            "joins": [
                {
                    "dataset_directory": "dataset/time",
                    "join_on": ["Date", "Date"],
                    "as": "time"
                },
                {
                    "dataset_directory": "dataset/retail/store",
                    "join_on": ["Store", "Store"],
                    "as": "store"
                }
            ]
        }
    }


@pytest.fixture()
def test_fd_retail_2(test_bucket, test_fd_retail):
    test_fd = test_fd_retail
    test_fd["forecast_definition"].update(
        {"dataset_directory": "{}/{}".format(test_bucket, test_fd["forecast_definition"]["dataset_directory"])})
    for join in test_fd["forecast_definition"]["joins"]:
        join.update({"dataset_directory": "{}/{}".format(test_bucket, join["dataset_directory"])})
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


@pytest.fixture()
def test_df_retail_sales():
    df = pd.DataFrame([[1, 5, "2015-07-31", 5263, 555, 1, 1, "z", 1],
                       [1, 4, "2015-07-30", 5020, 546, 1, 1, "z", 1],
                       [1, 3, "2015-07-29", 4782, 523, 1, 1, "z", 1],
                       [1, 2, "2015-07-28", 5011, 560, 1, 1, "z", 1],
                       [1, 1, "2015-07-27", 6102, 612, 1, 1, "z", 1],
                       [1, 7, "2015-07-26", 0, 0, 0, 0, "z", 0],
                       [1, 6, "2015-07-25", 4364, 500, 1, 0, "z", 0],
                       [1, 5, "2015-07-24", 3706, 459, 1, 0, "z", 0],
                       [1, 4, "2015-07-23", 3769, 503, 1, 0, "z", 0],
                       [1, 3, "2015-07-22", 3464, 463, 1, 0, "z", 0],
                       [1, 2, "2015-07-21", 3558, 469, 1, 0, "z", 0],
                       [1, 1, "2015-07-20", 4395, 526, 1, 0, "z", 0],
                       [1, 7, "2015-07-19", 0, 0, 0, 0, "z", 0],
                       [1, 6, "2015-07-18", 4406, 512, 1, 0, "z", 0],
                       [1, 5, "2015-07-17", 4852, 519, 1, 1, "z", 0],
                       [1, 4, "2015-07-16", 4427, 517, 1, 1, "z", 0],
                       [1, 3, "2015-07-15", 4767, 550, 1, 1, "z", 0],
                       [1, 2, "2015-07-14", 5042, 544, 1, 1, "z", 0],
                       [1, 1, "2015-07-13", 5054, 553, 1, 1, "z", 0],
                       [1, 7, "2015-07-12", 0, 0, 0, 0, "z", 0],
                       [1, 6, "2015-07-11", 3530, 441, 1, 0, "z", 0],
                       [1, 5, "2015-07-10", 3808, 449, 1, 0, "z", 0],
                       [1, 4, "2015-07-09", 3897, 480, 1, 0, "z", 0],
                       [1, 3, "2015-07-08", 3797, 485, 1, 0, "z", 0],
                       [1, 2, "2015-07-07", 3650, 485, 1, 0, "z", 0]])
    df.columns = ["Store", "DayOfWeek", "Date", "Sales", "Customers", "Open", "Promo",
                  "StateHoliday", "SchoolHoliday"]
    return df


@pytest.fixture()
def test_df_retail_stores():
    df = pd.DataFrame([[1.0, "c", "a", 1270.0, 9.0, 2008, 0.0, np.NaN, np.NaN, None]])
    df.columns = ["Store", "StoreType", "Assortment", "CompetitionDistance",
                  "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2",
                  "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]
    return df


@pytest.fixture()
def test_df_retail_time():
    df = pd.DataFrame([[Timestamp('2015-07-01 00:00:00'), 7, 1, 2015, 2, False, 'None', 27, 78708, 0, 1],
                       [Timestamp('2015-07-02 00:00:00'), 7, 2, 2015, 3, False, 'None', 27, 78709, 0, 2],
                       [Timestamp('2015-07-03 00:00:00'), 7, 3, 2015, 4, True, 'July 4th', 27, 78710, 0, 3],
                       [Timestamp('2015-07-04 00:00:00'), 7, 4, 2015, 5, False, 'None', 27, 78711, 0, 4],
                       [Timestamp('2015-07-05 00:00:00'), 7, 5, 2015, 6, False, 'None', 27, 78712, 0, 5],
                       [Timestamp('2015-07-06 00:00:00'), 7, 6, 2015, 0, False, 'None', 28, 78713, 0, 6],
                       [Timestamp('2015-07-07 00:00:00'), 7, 7, 2015, 1, False, 'None', 28, 78714, 0, 7],
                       [Timestamp('2015-07-08 00:00:00'), 7, 8, 2015, 2, False, 'None', 28, 78715, 0, 8],
                       [Timestamp('2015-07-09 00:00:00'), 7, 9, 2015, 3, False, 'None', 28, 78716, 0, 9],
                       [Timestamp('2015-07-10 00:00:00'), 7, 10, 2015, 4, False, 'None', 28, 78717, 0, 10],
                       [Timestamp('2015-07-11 00:00:00'), 7, 11, 2015, 5, False, 'None', 28, 78718, 0, 11],
                       [Timestamp('2015-07-12 00:00:00'), 7, 12, 2015, 6, False, 'None', 28, 78719, 0, 12],
                       [Timestamp('2015-07-13 00:00:00'), 7, 13, 2015, 0, False, 'None', 29, 78720, 0, 13],
                       [Timestamp('2015-07-14 00:00:00'), 7, 14, 2015, 1, False, 'None', 29, 78721, 0, 14],
                       [Timestamp('2015-07-15 00:00:00'), 7, 15, 2015, 2, False, 'None', 29, 78722, 0, 15],
                       [Timestamp('2015-07-16 00:00:00'), 7, 16, 2015, 3, False, 'None', 29, 78723, 0, 16],
                       [Timestamp('2015-07-17 00:00:00'), 7, 17, 2015, 4, False, 'None', 29, 78724, 0, 17],
                       [Timestamp('2015-07-18 00:00:00'), 7, 18, 2015, 5, False, 'None', 29, 78725, 0, 18],
                       [Timestamp('2015-07-19 00:00:00'), 7, 19, 2015, 6, False, 'None', 29, 78726, 0, 19],
                       [Timestamp('2015-07-20 00:00:00'), 7, 20, 2015, 0, False, 'None', 30, 78727, 0, 20],
                       [Timestamp('2015-07-21 00:00:00'), 7, 21, 2015, 1, False, 'None', 30, 78728, 0, 21],
                       [Timestamp('2015-07-22 00:00:00'), 7, 22, 2015, 2, False, 'None', 30, 78729, 0, 22],
                       [Timestamp('2015-07-23 00:00:00'), 7, 23, 2015, 3, False, 'None', 30, 78730, 0, 23],
                       [Timestamp('2015-07-24 00:00:00'), 7, 24, 2015, 4, False, 'None', 30, 78731, 0, 24],
                       [Timestamp('2015-07-25 00:00:00'), 7, 25, 2015, 5, False, 'None', 30, 78732, 0, 25],
                       [Timestamp('2015-07-26 00:00:00'), 7, 26, 2015, 6, False, 'None', 30, 78733, 0, 26],
                       [Timestamp('2015-07-27 00:00:00'), 7, 27, 2015, 0, False, 'None', 31, 78734, 0, 27],
                       [Timestamp('2015-07-28 00:00:00'), 7, 28, 2015, 1, False, 'None', 31, 78735, 0, 28],
                       [Timestamp('2015-07-29 00:00:00'), 7, 29, 2015, 2, False, 'None', 31, 78736, 0, 29],
                       [Timestamp('2015-07-30 00:00:00'), 7, 30, 2015, 3, False, 'None', 31, 78737, 0, 30],
                       [Timestamp('2015-07-31 00:00:00'), 7, 31, 2015, 4, False, 'None', 31, 78738, 1, 31],
                       [Timestamp('2015-08-01 00:00:00'), 8, 1, 2015, 5, False, 'None', 31, 78739, 0, 1],
                       [Timestamp('2015-08-02 00:00:00'), 8, 2, 2015, 6, False, 'None', 31, 78740, 0, 2],
                       [Timestamp('2015-08-03 00:00:00'), 8, 3, 2015, 0, False, 'None', 32, 78741, 0, 3],
                       [Timestamp('2015-08-04 00:00:00'), 8, 4, 2015, 1, False, 'None', 32, 78742, 0, 4],
                       [Timestamp('2015-08-05 00:00:00'), 8, 5, 2015, 2, False, 'None', 32, 78743, 0, 5],
                       [Timestamp('2015-08-06 00:00:00'), 8, 6, 2015, 3, False, 'None', 32, 78744, 0, 6],
                       [Timestamp('2015-08-07 00:00:00'), 8, 7, 2015, 4, False, 'None', 32, 78745, 0, 7],
                       [Timestamp('2015-08-08 00:00:00'), 8, 8, 2015, 5, False, 'None', 32, 78746, 0, 8],
                       [Timestamp('2015-08-09 00:00:00'), 8, 9, 2015, 6, False, 'None', 32, 78747, 0, 9],
                       [Timestamp('2015-08-10 00:00:00'), 8, 10, 2015, 0, False, 'None', 33, 78748, 0, 10],
                       [Timestamp('2015-08-11 00:00:00'), 8, 11, 2015, 1, False, 'None', 33, 78749, 0, 11],
                       [Timestamp('2015-08-12 00:00:00'), 8, 12, 2015, 2, False, 'None', 33, 78750, 0, 12],
                       [Timestamp('2015-08-13 00:00:00'), 8, 13, 2015, 3, False, 'None', 33, 78751, 0, 13],
                       [Timestamp('2015-08-14 00:00:00'), 8, 14, 2015, 4, False, 'None', 33, 78752, 0, 14],
                       [Timestamp('2015-08-15 00:00:00'), 8, 15, 2015, 5, False, 'None', 33, 78753, 0, 15],
                       [Timestamp('2015-08-16 00:00:00'), 8, 16, 2015, 6, False, 'None', 33, 78754, 0, 16],
                       [Timestamp('2015-08-17 00:00:00'), 8, 17, 2015, 0, False, 'None', 34, 78755, 0, 17],
                       [Timestamp('2015-08-18 00:00:00'), 8, 18, 2015, 1, False, 'None', 34, 78756, 0, 18],
                       [Timestamp('2015-08-19 00:00:00'), 8, 19, 2015, 2, False, 'None', 34, 78757, 0, 19],
                       [Timestamp('2015-08-20 00:00:00'), 8, 20, 2015, 3, False, 'None', 34, 78758, 0, 20],
                       [Timestamp('2015-08-21 00:00:00'), 8, 21, 2015, 4, False, 'None', 34, 78759, 0, 21],
                       [Timestamp('2015-08-22 00:00:00'), 8, 22, 2015, 5, False, 'None', 34, 78760, 0, 22],
                       [Timestamp('2015-08-23 00:00:00'), 8, 23, 2015, 6, False, 'None', 34, 78761, 0, 23],
                       [Timestamp('2015-08-24 00:00:00'), 8, 24, 2015, 0, False, 'None', 35, 78762, 0, 24],
                       [Timestamp('2015-08-25 00:00:00'), 8, 25, 2015, 1, False, 'None', 35, 78763, 0, 25],
                       [Timestamp('2015-08-26 00:00:00'), 8, 26, 2015, 2, False, 'None', 35, 78764, 0, 26],
                       [Timestamp('2015-08-27 00:00:00'), 8, 27, 2015, 3, False, 'None', 35, 78765, 0, 27],
                       [Timestamp('2015-08-28 00:00:00'), 8, 28, 2015, 4, False, 'None', 35, 78766, 0, 28],
                       [Timestamp('2015-08-29 00:00:00'), 8, 29, 2015, 5, False, 'None', 35, 78767, 0, 29],
                       [Timestamp('2015-08-30 00:00:00'), 8, 30, 2015, 6, False, 'None', 35, 78768, 0, 30],
                       [Timestamp('2015-08-31 00:00:00'), 8, 31, 2015, 0, False, 'None', 36, 78769, 1, 31]])
    df.columns = ['Date', 'Month', 'Day', 'Year', 'Weekday', 'Holiday', 'HolidayType', 'WeekOfYear', 'T',
                  'LastDayOfMonth', 'DayOfMonth']
    return df
