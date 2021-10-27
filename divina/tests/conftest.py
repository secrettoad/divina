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
def test_model_1(test_df_1, random_state, test_fd_1):
    params = [31.835024241839548, -0.7787925146603903, 8.57527111840826]
    intercept = -8.575161145107172
    features = ['b', 'b_(5, 10]', 'b_(15, inf]']

    model = LinearRegression()
    model.fit(
        ddf.from_pandas(pd.DataFrame([np.array(params) + c for c in range(0, 2)]), npartitions=1).to_dask_array(
            lengths=True),
        ddf.from_pandas(pd.Series([intercept, intercept]), npartitions=1).to_dask_array(lengths=True))
    model.coef_ = np.array(params)
    model.intercept_ = intercept
    model._coef = np.array(params + [intercept])
    return (model, {"params": {f: c for f, c in zip(features, [intercept] + params)}})


@pytest.fixture()
def test_model_retail(test_df_1, random_state, test_fd_1):
    params = [7.590629527751595, -0.4168572808786085, 0.010925907309850878, 0.2799784009152658, 0.41685771462138027,
              0.559612229671561, 0.6053793270928081, 0.7243317875555008, 0.5257454418529863, -3.706648609469239,
              0.5727582262743428, 0.041242474438584156, 0.48795463528528316, 0.5346672084741427, 0.751490274495951,
              0.263377351712863, 0.23346362705895096, 0.5778595049653592, -3.3000175997266337, 0.6876410670504797,
              0.6322753860791753, 0.4828332692767577, 0.4584858218915188, -3.4403200753143013, 0.588939024932829,
              0.2667369612563883, 0.5425191970467125, 0.48409241686815374, 0.7163856760343319, 0.5067036598551107,
              0.6339288839409306, 0.7108781138655268, -3.3687035041613527, 0.5075748621530489, -0.03047872449177162,
              -0.026704766769665735, 0.2799784009153089]
    intercept = -0.1019028186184816
    features = ['Open', 'Promo', 'SchoolHoliday', 'DayOfWeek_7.0', 'DayOfWeek_2.0', 'DayOfWeek_4.0', 'DayOfWeek_6.0',
                'DayOfWeek_1.0', 'DayOfWeek_5.0', 'DayOfWeek_3.0', 'DayOfMonth_19.0', 'DayOfMonth_21.0',
                'DayOfMonth_23.0', 'DayOfMonth_25.0', 'DayOfMonth_27.0', 'DayOfMonth_28.0', 'DayOfMonth_7.0',
                'DayOfMonth_10.0', 'DayOfMonth_13.0', 'DayOfMonth_16.0', 'DayOfMonth_20.0', 'DayOfMonth_22.0',
                'DayOfMonth_24.0', 'DayOfMonth_26.0', 'DayOfMonth_29.0', 'DayOfMonth_8.0', 'DayOfMonth_9.0',
                'DayOfMonth_11.0', 'DayOfMonth_12.0', 'DayOfMonth_14.0', 'DayOfMonth_15.0', 'DayOfMonth_17.0',
                'DayOfMonth_18.0', 'WeekOfYear_29.0', 'WeekOfYear_30.0', 'WeekOfYear_31.0', 'WeekOfYear_28.0']

    model = LinearRegression()
    model.fit(
        ddf.from_pandas(pd.DataFrame([np.array(params) + c for c in range(0, 2)]), npartitions=1).to_dask_array(
            lengths=True),
        ddf.from_pandas(pd.Series([intercept, intercept]), npartitions=1).to_dask_array(lengths=True))
    model.coef_ = np.array(params)
    model.intercept_ = intercept
    model._coef = np.array(params + [intercept])
    return (model, {"params": {f: c for f, c in zip(features, [intercept] + params)}})


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
    features = [['b', 'b_(5, 10]', 'b_(15, inf]'], ['b', 'b_(5, 10]', 'b_(15, inf]'], ['b', 'b_(5, 10]', 'b_(15, inf]'],
                ['b', 'b_(5, 10]', 'b_(15, inf]'], ['b', 'b_(5, 10]', 'b_(15, inf]']]
    seeds = range(random_state, random_state + test_fd_1['forecast_definition']['bootstrap_sample'])
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
        bootstrap_models[seed] = (model, {"params": {_f: c for _f, c in zip(f, [i] + p)}})
    return bootstrap_models


@pytest.fixture()
def test_bootstrap_models_retail(test_df_1, random_state, test_fd_1):
    params = [[8.056567343707801, -0.14506670832768434, -0.23232275952547266, 0.20150285165630455, 0.14507498439276545,
               0.27713364973522986, 0.10221104900967637, 0.31536809193941906, 0.2559513552431588, -3.647847389506965,
               0.24163391359012484, 0.04678182659155821, 0.17623818052083737, 0.2435980816885483, 0.18484679078064845,
               0.10926352998806904, -0.020868422031112856, 0.6447880073419305, -0.1470886134243738,
               -0.11629242568466953, 0.23033142252793867, 0.2432398543645691, -0.052418195664897096,
               0.016831361903957214, 0.20470484882999093, 0.10976261331097335, 0.6506371953867083, 0.7309318379046927,
               -3.647847389506965, 0.39268788536980687, -0.27278010336450864, 0.09945416171946965, 0.20150285165633255],
              [7.393789969515538, -0.5124427262995791, -0.08463509449707678, 0.3729452233136588, 0.512482198394133,
               0.700251285100012, 0.7432343973751843, 0.8642724320337655, 0.5721274407425915, -3.5947822924566806,
               0.5804188589797001, 0.22023738533519752, 0.6244587928831947, 0.8909260462901273, 0.36338890618482605,
               0.23564072668878885, 0.6098628757819881, -3.2306727274108176, 0.8750282485588204, 0.7173164230322971,
               -3.2548720060739087, 0.6468777240369674, 0.3955350196607907, 0.7008318773375861, 0.5288791098533079,
               0.7410065451127075, 0.4995481433768048, 0.8642786287759757, -3.1004973227017407, 0.6728379510518601,
               -0.12039755066249616, -0.09652901345282944, 0.37294522331366026],
              [7.373882306359122, -0.5012311643844884, -0.010309484921238118, 0.29809637779981607, 0.5012290452763202,
               0.7240970671453367, 0.701156677178618, 0.7735826006036559, 0.6156584281412932, -3.618723007857756,
               0.73312439759998, 0.07810173507541884, 0.45547351794606983, 0.6519865298567203, 0.9257017037648314,
               0.29809637779981607, 0.7886601211450533, -3.04333665995808, 0.8000492449455986, 0.6724194445407347,
               0.5436078203751433, 0.4301198776195328, -3.337138895210775, 0.569240384830692, 0.7574571093441058,
               0.6892663022412002, 0.6867895080402127, 0.7674190413801395, -3.269454831041309, 0.5355087926577404,
               0.05803742347607925, 0.05121413852980488, 0.2980963777998064],
              [7.710389533072281, -0.43227397293312364, 0.2353534850658354, 0.1979529587609676, 0.43228082130422735,
               0.42416578608176664, 0.46185329853401585, 0.6261674418846083, 0.3947978959571063, -3.6787499819307157,
               0.4017249835338047, 0.5506298713988833, 0.6144498021665386, 0.7784557872093627, 0.1920004237992822,
               0.1540607917926774, 0.5938229421076817, -3.463253104832229, 0.11521627310634114, 0.5537318795106152,
               -3.4854939167571706, 0.39934426636652903, 0.18181977436502267, 0.628049583287984, 0.5019491540789822,
               0.6916020048263761, 0.4171891181546088, 0.11538258056744165, 0.2614036983649554, 0.19972848625106238,
               0.20751624180872852, -0.13590017311133695, 0.19795295876096822],
              [7.347995761069305, -0.5306287923508407, 0.04733673002955241, 0.3209527028227726, 0.5306273909568222,
               0.6546546224145695, 0.7341850480214894, 0.8280762774334178, 0.6717468402870133, -3.588280980897007,
               0.7205369406795783, 0.23451003022802092, 0.7670480251225463, 0.26100074844643906, 0.32202159622159676,
               0.6566744742530832, -3.1992529110984687, 0.8591872664086128, 0.8136082902750973, 0.6711353284019933,
               0.655390769272413, -3.1946981526454077, 0.6811744050044131, 0.2728499653511513, 0.6065154613320831,
               0.5731849734526444, 0.8280762774334178, 0.4993203764774724, 0.8670311184508511, -3.174799782823085,
               -0.10297532899338556, -0.10798615546350797, 0.3209527028228095]]
    intercepts = [0.0663407344762792, -0.03477010598530712, -0.24238480956883526, -0.1455366328662099,
                  -0.008186968767883647]
    features = [['Open', 'Promo', 'SchoolHoliday', 'DayOfWeek_7.0', 'DayOfWeek_2.0', 'DayOfWeek_4.0', 'DayOfWeek_6.0',
                 'DayOfWeek_1.0', 'DayOfWeek_5.0', 'DayOfWeek_3.0', 'DayOfMonth_19.0', 'DayOfMonth_23.0',
                 'DayOfMonth_25.0', 'DayOfMonth_27.0', 'DayOfMonth_28.0', 'DayOfMonth_7.0', 'DayOfMonth_13.0',
                 'DayOfMonth_20.0', 'DayOfMonth_22.0', 'DayOfMonth_26.0', 'DayOfMonth_29.0', 'DayOfMonth_8.0',
                 'DayOfMonth_9.0', 'DayOfMonth_11.0', 'DayOfMonth_12.0', 'DayOfMonth_14.0', 'DayOfMonth_15.0',
                 'DayOfMonth_17.0', 'DayOfMonth_18.0', 'WeekOfYear_29.0', 'WeekOfYear_30.0', 'WeekOfYear_31.0',
                 'WeekOfYear_28.0'],
                ['Open', 'Promo', 'SchoolHoliday', 'DayOfWeek_7.0', 'DayOfWeek_2.0', 'DayOfWeek_4.0', 'DayOfWeek_6.0',
                 'DayOfWeek_1.0', 'DayOfWeek_5.0', 'DayOfWeek_3.0', 'DayOfMonth_19.0', 'DayOfMonth_21.0',
                 'DayOfMonth_25.0', 'DayOfMonth_27.0', 'DayOfMonth_28.0', 'DayOfMonth_7.0', 'DayOfMonth_10.0',
                 'DayOfMonth_16.0', 'DayOfMonth_20.0', 'DayOfMonth_24.0', 'DayOfMonth_26.0', 'DayOfMonth_29.0',
                 'DayOfMonth_8.0', 'DayOfMonth_9.0', 'DayOfMonth_11.0', 'DayOfMonth_12.0', 'DayOfMonth_14.0',
                 'DayOfMonth_17.0', 'DayOfMonth_18.0', 'WeekOfYear_29.0', 'WeekOfYear_30.0', 'WeekOfYear_31.0',
                 'WeekOfYear_28.0'],
                ['Open', 'Promo', 'SchoolHoliday', 'DayOfWeek_7.0', 'DayOfWeek_2.0', 'DayOfWeek_4.0', 'DayOfWeek_6.0',
                 'DayOfWeek_1.0', 'DayOfWeek_5.0', 'DayOfWeek_3.0', 'DayOfMonth_19.0', 'DayOfMonth_21.0',
                 'DayOfMonth_23.0', 'DayOfMonth_25.0', 'DayOfMonth_27.0', 'DayOfMonth_7.0', 'DayOfMonth_10.0',
                 'DayOfMonth_13.0', 'DayOfMonth_16.0', 'DayOfMonth_20.0', 'DayOfMonth_22.0', 'DayOfMonth_24.0',
                 'DayOfMonth_26.0', 'DayOfMonth_8.0', 'DayOfMonth_12.0', 'DayOfMonth_14.0', 'DayOfMonth_15.0',
                 'DayOfMonth_17.0', 'DayOfMonth_18.0', 'WeekOfYear_29.0', 'WeekOfYear_30.0', 'WeekOfYear_31.0',
                 'WeekOfYear_28.0'],
                ['Open', 'Promo', 'SchoolHoliday', 'DayOfWeek_7.0', 'DayOfWeek_2.0', 'DayOfWeek_4.0', 'DayOfWeek_6.0',
                 'DayOfWeek_1.0', 'DayOfWeek_5.0', 'DayOfWeek_3.0', 'DayOfMonth_21.0', 'DayOfMonth_23.0',
                 'DayOfMonth_25.0', 'DayOfMonth_27.0', 'DayOfMonth_28.0', 'DayOfMonth_7.0', 'DayOfMonth_10.0',
                 'DayOfMonth_16.0', 'DayOfMonth_20.0', 'DayOfMonth_24.0', 'DayOfMonth_26.0', 'DayOfMonth_29.0',
                 'DayOfMonth_8.0', 'DayOfMonth_9.0', 'DayOfMonth_11.0', 'DayOfMonth_12.0', 'DayOfMonth_14.0',
                 'DayOfMonth_15.0', 'DayOfMonth_18.0', 'WeekOfYear_29.0', 'WeekOfYear_30.0', 'WeekOfYear_31.0',
                 'WeekOfYear_28.0'],
                ['Open', 'Promo', 'SchoolHoliday', 'DayOfWeek_7.0', 'DayOfWeek_2.0', 'DayOfWeek_4.0', 'DayOfWeek_6.0',
                 'DayOfWeek_1.0', 'DayOfWeek_5.0', 'DayOfWeek_3.0', 'DayOfMonth_19.0', 'DayOfMonth_21.0',
                 'DayOfMonth_27.0', 'DayOfMonth_28.0', 'DayOfMonth_7.0', 'DayOfMonth_10.0', 'DayOfMonth_13.0',
                 'DayOfMonth_16.0', 'DayOfMonth_20.0', 'DayOfMonth_22.0', 'DayOfMonth_24.0', 'DayOfMonth_26.0',
                 'DayOfMonth_29.0', 'DayOfMonth_8.0', 'DayOfMonth_9.0', 'DayOfMonth_11.0', 'DayOfMonth_12.0',
                 'DayOfMonth_15.0', 'DayOfMonth_17.0', 'WeekOfYear_29.0', 'WeekOfYear_30.0', 'WeekOfYear_31.0',
                 'WeekOfYear_28.0']]
    seeds = range(random_state, random_state + test_fd_1['forecast_definition']['bootstrap_sample'])
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
        bootstrap_models[seed] = (model, {"params": {_f: c for _f, c in zip(f, [i] + p)}})
    return bootstrap_models


@pytest.fixture()
def test_params_2(test_model_1):
    return {"params": {c: test_model_1[1]['params'][c] + 1 for c in test_model_1[1]['params']}}


@pytest.fixture()
def test_metrics_1():
    return {'splits': {'1970-01-01 00:00:07': {'time_horizons': {'1': {'mae': 14.686136955152564}}}}}


@pytest.fixture()
def test_metrics_retail():
    return {'splits': {'2015-07-18': {'time_horizons': {'2': {'mae': 4162.277118203362}}}}}


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
        [[Timestamp('2015-07-19 00:00:00'), 1.8317264098620485], [Timestamp('2015-07-21 00:00:00'), 7.938372961307323],
         [Timestamp('2015-07-23 00:00:00'), 10.554629030437354], [Timestamp('2015-07-25 00:00:00'), 13.803328831072312],
         [Timestamp('2015-07-27 00:00:00'), 16.952544389150717], [Timestamp('2015-07-28 00:00:00'), 16.92227823307985],
         [Timestamp('2015-07-30 00:00:00'), 19.218858622600237], [Timestamp('2015-07-20 00:00:00'), 8.001938667333889],
         [Timestamp('2015-07-22 00:00:00'), 9.093323066059384], [Timestamp('2015-07-24 00:00:00'), 11.456502555503626],
         [Timestamp('2015-07-26 00:00:00'), 8.216953797283812], [Timestamp('2015-07-29 00:00:00'), 17.870771128084016],
         [Timestamp('2015-07-31 00:00:00'), 20.473707377235492], [Timestamp('2015-07-18 00:00:00'), 7.498020363556597]])
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
    df = pd.DataFrame([[Timestamp('2015-07-21 00:00:00'), 7.938372961307323, 10.05660735992933, 5.864523315639474],
                       [Timestamp('2015-07-23 00:00:00'), 10.554629030437354, 12.762422159223206, 9.937192645555214],
                       [Timestamp('2015-07-25 00:00:00'), 13.803328831072312, 16.008575798354855, 9.652716573065845],
                       [Timestamp('2015-07-27 00:00:00'), 16.95254438914708, 19.700064530923555, 15.878979830766184],
                       [Timestamp('2015-07-28 00:00:00'), 16.92227823307985, 19.853075073217042, 15.955625570939446],
                       [Timestamp('2015-07-30 00:00:00'), 19.218858622600237, 22.26022209099392, 18.96366030318677],
                       [Timestamp('2015-07-22 00:00:00'), 9.093323066059384, 12.0369650356879, 6.941501152124715],
                       [Timestamp('2015-07-24 00:00:00'), 11.456502555503626, 14.5063345079476, 9.295135387800357],
                       [Timestamp('2015-07-26 00:00:00'), 8.216953797283812, 10.987461008674131, -2.094508001193674],
                       [Timestamp('2015-07-29 00:00:00'), 17.870771128084016, 21.64054314007808, 16.599755525935592],
                       [Timestamp('2015-07-31 00:00:00'), 20.473707377235492, 24.33391344847462, 19.408378340531314]])
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
            "include_features": ['Store', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek',
                                 'LastDayOfMonth', 'DayOfMonth', 'WeekOfYear', 'MonthOfYear'],
            "time_validation_splits": ["2015-07-18"],
            "forecast_end": "2016-01-01",
            "bootstrap_sample": 5,
            "signal_dimensions": ['Store'],
            "time_horizons": [2],
            "forecast_freq": 'D',
            "encode_features": ['StateHoliday', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'MonthOfYear'],
            "dataset_directory": "dataset/retail/sales2",
            "link_function": "log",
            "confidence_intervals": [90, 10],
            "joins": [
                {
                    "dataset_directory": "dataset/time",
                    "join_on": ("Date", "Date"),
                    "as": "time"
                },
                {
                    "dataset_directory": "dataset/retail/store",
                    "join_on": ("Store", "Store"),
                    "as": "store"
                }
            ]
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
    df = pd.DataFrame([['2015-07-07', 7, 7, 2015, 1, False, None, 78714, 0, 28, 7, 7],
                       ['2015-07-08', 7, 8, 2015, 2, False, None, 78715, 0, 28, 7, 8],
                       ['2015-07-09', 7, 9, 2015, 3, False, None, 78716, 0, 28, 7, 9],
                       ['2015-07-10', 7, 10, 2015, 4, False, None, 78717, 0, 28, 7, 10],
                       ['2015-07-11', 7, 11, 2015, 5, False, None, 78718, 0, 28, 7, 11],
                       ['2015-07-12', 7, 12, 2015, 6, False, None, 78719, 0, 28, 7, 12],
                       ['2015-07-13', 7, 13, 2015, 0, False, None, 78720, 0, 29, 7, 13],
                       ['2015-07-14', 7, 14, 2015, 1, False, None, 78721, 0, 29, 7, 14],
                       ['2015-07-15', 7, 15, 2015, 2, False, None, 78722, 0, 29, 7, 15],
                       ['2015-07-16', 7, 16, 2015, 3, False, None, 78723, 0, 29, 7, 16],
                       ['2015-07-17', 7, 17, 2015, 4, False, None, 78724, 0, 29, 7, 17],
                       ['2015-07-18', 7, 18, 2015, 5, False, None, 78725, 0, 29, 7, 18],
                       ['2015-07-19', 7, 19, 2015, 6, False, None, 78726, 0, 29, 7, 19],
                       ['2015-07-20', 7, 20, 2015, 0, False, None, 78727, 0, 30, 7, 20],
                       ['2015-07-21', 7, 21, 2015, 1, False, None, 78728, 0, 30, 7, 21],
                       ['2015-07-22', 7, 22, 2015, 2, False, None, 78729, 0, 30, 7, 22],
                       ['2015-07-23', 7, 23, 2015, 3, False, None, 78730, 0, 30, 7, 23],
                       ['2015-07-24', 7, 24, 2015, 4, False, None, 78731, 0, 30, 7, 24],
                       ['2015-07-25', 7, 25, 2015, 5, False, None, 78732, 0, 30, 7, 25],
                       ['2015-07-26', 7, 26, 2015, 6, False, None, 78733, 0, 30, 7, 26],
                       ['2015-07-27', 7, 27, 2015, 0, False, None, 78734, 0, 31, 7, 27],
                       ['2015-07-28', 7, 28, 2015, 1, False, None, 78735, 0, 31, 7, 28],
                       ['2015-07-29', 7, 29, 2015, 2, False, None, 78736, 0, 31, 7, 29],
                       ['2015-07-30', 7, 30, 2015, 3, False, None, 78737, 0, 31, 7, 30],
                       ['2015-07-31', 7, 31, 2015, 4, False, None, 78738, 1, 31, 7, 31]])
    df.columns = ['Date', 'Month', 'Day', 'Year', 'Weekday', 'Holiday', 'HolidayType',
                  'T', 'LastDayOfMonth', 'WeekOfYear', 'MonthOfYear', 'DayOfMonth']
    return df
