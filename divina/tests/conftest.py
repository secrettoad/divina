import json
import os
import pathlib
import shutil
from unittest.mock import patch

import dask.dataframe as ddf
import fsspec
import numpy as np
import pandas as pd
import pytest
from dask.distributed import Client
from dask_ml.linear_model import LinearRegression
from pandas import Timestamp
from pykube import Pod, PersistentVolume
import time
from pipeline.model import GLM, EWMA
from pipeline.pipeline import Pipeline, PipelineValidation, ValidationSplit, BoostValidation, CausalValidation, \
    Validation
from sklearn.base import BaseEstimator
import kfp


@pytest.fixture()
def random_state():
    return 11


@pytest.fixture()
def test_model_1(test_df_1, random_state, test_pipeline_1):
    params = [0.33754904224551174, 3.898442928579839, 3.898442928579839, 2.1651306296852373, 2.0252942534730707,
              3.307291666666666]
    intercept = -14.649630725378326
    fit_indices = [0, 1, 2, 3, 4, 5]

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
    model.fit_indices = fit_indices
    return model


@pytest.fixture()
def test_params_1(test_model_1):
    return test_model_1[1]


@pytest.fixture()
def test_bootstrap_models(random_state, test_pipeline_1):
    params = [
        [5.807879880277484, 5.996090135921985, -5.996103319528735],
         [6.405775482139099, 8.275877252257798, -8.275879348428148],
         [2.450067447370109, -0.04323536076015728, 0.043236724281565886],
         [6.002238892041233, 6.9669916158632565, -6.966994085850243],
         [7.518383561688452, 8.25293678195981, -8.252982252663998]
    ]
    intercepts = [-15.632378932373904, -18.447719997424628, -2.8743744329911856, -16.331371393981936,
                  -20.651160098319018]
    indices = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
    states = range(
        random_state,
        random_state + test_pipeline_1.bootstrap_sample,
    )
    bootstrap_models = []

    for j, i, p, f, state in zip(
            range(0, len(states)), intercepts, params, indices, states
    ):
        model = GLM(link_function='log')
        model.linear_model.fit(
            ddf.from_pandas(
                pd.DataFrame([np.array(params[j]) + c for c in range(0, len(states))]),
                npartitions=1,
            ).to_dask_array(lengths=True),
            ddf.from_pandas(pd.Series(intercepts), npartitions=1).to_dask_array(
                lengths=True
            ),
        )
        model.linear_model.coef_ = np.array(p)
        model.linear_model.intercept_ = i
        model.linear_model._coef = np.array(p + [i])
        model.fit_indices = f
        bootstrap_models.append(model)
    return bootstrap_models


@pytest.fixture()
def test_horizons():
    return range(3, 5)


@pytest.fixture()
def test_boost_models(test_df_1, test_pipeline_1, test_horizons):
    boost_models = {h: EWMA() for h in test_horizons}
    for h in boost_models:
        boost_models[h].window = test_pipeline_1.boost_window
    return boost_models


@pytest.fixture()
def test_params_2(test_model_1):
    return test_model_1[1]


@pytest.fixture()
def test_metrics_1():
    return {'mse': 286.2579120259681}


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
    df = pd.DataFrame()
    df['a'] = pd.to_datetime(['1970-01-01 00:00:01', '1970-01-01 00:00:02',
                              '1970-01-01 00:00:03', '1970-01-01 00:00:04',
                              '1970-01-01 00:00:05', '1970-01-01 00:00:06',
                              '1970-01-01 00:00:07', '1970-01-01 00:00:08',
                              '1970-01-01 00:00:09', '1970-01-01 00:00:10'])
    df['y_hat'] = [0.8921926222729465, 0.8921926222729465, 0.8921926222729465, 14.99863188613795, 0.04329555074033657,
                   0.04329555074033657, 21.56055802208144, 21.56055802208144, 21.56055802208144, 101.79384189509244]
    df = df.set_index('a')
    return ddf.from_pandas(df, npartitions=2)['y_hat']


@pytest.fixture()
def test_pipeline_name():
    return 'test-pipeline'


@pytest.fixture()
def test_pipeline_root(test_pipeline_name, test_bucket):
    return 's3://{}/{}'.format(test_bucket, test_pipeline_name)


@pytest.fixture()
def test_pipeline_1():
    return Pipeline(
        time_index="a",
        target="c",
        validation_splits=["1970-01-01 00:00:07"],
        validate_start="1970-01-01 00:00:01",
        validate_end="1970-01-01 00:00:09",
        frequency="S",
        confidence_intervals=[90],
        bootstrap_sample=5,
        bin_features={"b": [5, 10, 15]},
        time_horizons=[1],
        boost_window=5)


@pytest.fixture()
def test_simulate_end():
    return "1970-01-01 00:00:14"


@pytest.fixture()
def test_simulate_start():
    return "1970-01-01 00:00:11"


@pytest.fixture()
def test_pipeline_2(test_pipeline_root, test_horizons):
    return Pipeline(time_index="a",
                    target="c",
                    validation_splits=["1970-01-01 00:00:07"],
                    validate_start="1970-01-01 00:00:01",
                    validate_end="1970-01-01 00:00:09",
                    frequency="S",
                    confidence_intervals=[90],
                    bootstrap_sample=5,
                    random_seed=11,
                    bin_features={"b": [5, 10, 15]},
                    time_horizons=test_horizons,
                    pipeline_root=test_pipeline_root,
                    target_dimensions=['d', 'e'],
                    boost_model_params={'alpha': 0.8},
                    boost_window=5)


@pytest.fixture()
def test_scenarios():
    return [{"b": x} for x in [0, 1]]


@pytest.fixture()
def test_pipelines_quickstart():
    eds = {}
    for file in sorted(
            os.listdir(
                pathlib.Path(
                    pathlib.Path(__file__).parent.parent.parent,
                    "docs_src/_static/pipeline_definitions",
                )
            )
    ):
        with open(
                pathlib.Path(
                    pathlib.Path(__file__).parent.parent.parent,
                    "docs_src/_static/pipeline_definitions",
                    file,
                )
        ) as f:
            eds[file.split(".")[0]] = json.load(f)
    return eds


@pytest.fixture()
def test_pipeline_3(test_bucket, test_pipeline_1):
    test_pipeline = test_pipeline_1
    # test_pipeline["pipeline_definition"].update(
    #    {"data_path": "{}/dataset/test1".format(test_bucket)}
    # )
    return test_pipeline


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
    for c in ['e', 'd']:
        R = np.random.RandomState(11)
        df[c] = R.randint(1, 3, df.shape[0])
    return ddf.from_pandas(df, npartitions=2)


@pytest.fixture()
def test_df_1():
    df = pd.DataFrame(
        [[Timestamp('1970-01-01 00:00:01'), 2.0, 1.5, 1.5, 12.0, 1, 0, 0, 0],
         [Timestamp('1970-01-01 00:00:02'), 2.0, 1.5, 1.5, 0.0, 1, 0, 0, 0],
         [Timestamp('1970-01-01 00:00:03'), 2.0, 1.5, 1.5, 0.0, 1, 0, 0, 0],
         [Timestamp('1970-01-01 00:00:04'), 5.0, 1.75, 1.75, 24.0, 0, 1, 0, 0],
         [Timestamp('1970-01-01 00:00:05'), 5.0, 1.0, 1.0, 18.0, 0, 1, 0, 0],
         [Timestamp('1970-01-01 00:00:06'), 5.0, 1.0, 1.0, 6.0, 0, 1, 0, 0],
         [Timestamp('1970-01-01 00:00:07'), 8.0, 1.6666666666666667, 1.6666666666666667, 54.0, 0, 1, 0, 0],
         [Timestamp('1970-01-01 00:00:08'), 8.0, 1.6666666666666667, 1.6666666666666667, 0.0, 0, 1, 0, 0],
         [Timestamp('1970-01-01 00:00:09'), 8.0, 1.6666666666666667, 1.6666666666666667, 0.0, 0, 1, 0, 0],
         [Timestamp('1970-01-01 00:00:10'), 11.0, 1.5714285714285714, 1.5714285714285714, 84.0, 0, 0, 1, 0]]
    )
    df.columns = ['a', 'b', 'e', 'd', 'c', 'b_(-inf, 5]', 'b_(5, 10]', 'b_(10, 15]', 'b_(15, inf]']
    return ddf.from_pandas(df, npartitions=2).set_index('a')


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
        R = np.random.RandomState(11)
        df[c] = R.randint(1, 3, df.shape[0])
    return ddf.from_pandas(df, npartitions=2)


@pytest.fixture
def test_boosted_predictions():
    df = pd.DataFrame(
        [[45.58413137156656], [60.7895464849905], [13.859510463526139], [28.79254702198098], [24.051514028073512],
         [13.313258838703597], [44.942921520368664], [61.145193973857204], [15.233716818422364], [25.008343275203817],
         [23.468016925036036], [16.312445897505494], [42.58681555603796], [63.87539354585601], [11.786678091145998],
         [27.588968494855635], [21.468108056979702], [10.418534142054174], [40.77127346734583], [60.29947992722473],
         [9.813262777350197], [23.702105865908596], [20.665251630170108], [9.545864112913312], [38.03477476759858],
         [56.52041334521233]])
    df.columns = [0]
    return ddf.from_pandas(df, npartitions=2)


@pytest.fixture
def test_causal_predictions():
    df = pd.DataFrame(
        [[38.03477476759858], [56.52041334521233], [7.532516315578564], [20.18657200300599], [16.99939293790017],
         [5.620208876515074], [38.03477476759858], [56.52041334521233], [7.532516315578564], [20.18657200300599],
         [16.99939293790017], [5.620208876515074], [38.03477476759858], [56.52041334521233], [7.532516315578564],
         [20.18657200300599], [16.99939293790017], [5.620208876515074], [38.03477476759858], [56.52041334521233],
         [7.532516315578564], [20.18657200300599], [16.99939293790017], [5.620208876515074], [38.03477476759858],
         [56.52041334521233]])
    df.columns = ['mean']
    return ddf.from_pandas(df, npartitions=2)


@pytest.fixture
def test_truth():
    df = pd.DataFrame([['1970-01-01 00:00:07__index__2__index__1', 48.0, 54.0, 0, 1],
                       ['1970-01-01 00:00:10__index__2__index__2', 77.0, 84.0, 0, 1],
                       ['1970-01-02 00:00:01__index__1__index__1', 8.0, 12.0, 1, 0],
                       ['1970-01-02 00:00:04__index__1__index__2', 20.0, 24.0, 0, 1],
                       ['1970-01-02 00:00:05__index__2__index__1', 15.0, 18.0, 0, 1],
                       ['1970-01-02 00:00:06__index__2__index__2', 5.0, 6.0, 1, 0],
                       ['1970-01-02 00:00:07__index__1__index__1', 48.0, 54.0, 0, 1],
                       ['1970-01-02 00:00:10__index__1__index__2', 77.0, 84.0, 0, 1],
                       ['1970-01-03 00:00:01__index__2__index__1', 8.0, 12.0, 1, 0],
                       ['1970-01-03 00:00:04__index__1__index__1', 20.0, 24.0, 0, 1],
                       ['1970-01-03 00:00:05__index__1__index__1', 15.0, 18.0, 0, 1],
                       ['1970-01-03 00:00:06__index__1__index__2', 5.0, 6.0, 1, 0],
                       ['1970-01-03 00:00:07__index__1__index__2', 48.0, 54.0, 0, 1],
                       ['1970-01-03 00:00:10__index__1__index__2', 77.0, 84.0, 0, 1],
                       ['1970-01-04 00:00:01__index__1__index__1', 8.0, 12.0, 1, 0],
                       ['1970-01-04 00:00:04__index__2__index__1', 20.0, 24.0, 0, 1],
                       ['1970-01-04 00:00:05__index__1__index__1', 15.0, 18.0, 0, 1],
                       ['1970-01-04 00:00:06__index__2__index__1', 5.0, 6.0, 1, 0],
                       ['1970-01-04 00:00:07__index__1__index__1', 48.0, 54.0, 0, 1],
                       ['1970-01-04 00:00:10__index__2__index__2', 77.0, 84.0, 0, 1],
                       ['1970-01-05 00:00:01__index__1__index__1', 8.0, 12.0, 1, 0],
                       ['1970-01-05 00:00:04__index__1__index__1', 20.0, 24.0, 0, 1],
                       ['1970-01-05 00:00:05__index__2__index__1', 15.0, 18.0, 0, 1],
                       ['1970-01-05 00:00:06__index__1__index__1', 5.0, 6.0, 1, 0],
                       ['1970-01-05 00:00:07__index__2__index__2', 48.0, 54.0, 0, 1],
                       ['1970-01-05 00:00:10__index__1__index__2', 77.0, 84.0, 0, 1]])
    df.columns = ['__target_dimension_index__', 'b', 'c', 'b_(5, 10]', 'b_(15, inf]']
    return ddf.from_pandas(df.set_index('__target_dimension_index__'), npartitions=2)


@pytest.fixture
def test_bootstrap_metrics():
    return [{'mse': 124.16190403858947},
            {'mse': 229.04037716654764},
            {'mse': 245.7282869286211},
            {'mse': 188.34572704380332}]


@pytest.fixture
def test_bootstrap_predictions():
    return [[[40.944542248313226], [62.83944297443861], [8.425711791823481], [19.804638098950782],
             [16.02965521513606], [6.160722061534648], [40.944542248313226], [62.83944297443861], [8.425711791823481],
             [19.804638098950782], [16.02965521513606], [6.160722061534648], [40.944542248313226],
             [62.83944297443861], [8.425711791823481], [19.804638098950782], [16.02965521513606], [6.160722061534648],
             [40.944542248313226], [62.83944297443861], [8.425711791823481], [19.804638098950782],
             [16.02965521513606], [6.160722061534648], [40.944542248313226], [62.83944297443861]],
            [[37.024804743734904], [54.75039203362197], [5.8903503351407185], [19.910444601774977],
             [16.854308862139277], [4.056668891359298], [37.024804743734904], [54.75039203362197],
             [5.8903503351407185], [19.910444601774977], [16.854308862139277], [4.056668891359298],
             [37.024804743734904], [54.75039203362197], [5.8903503351407185], [19.910444601774977],
             [16.854308862139277], [4.056668891359298], [37.024804743734904], [54.75039203362197],
             [5.8903503351407185], [19.910444601774977], [16.854308862139277], [4.056668891359298],
             [37.024804743734904], [54.75039203362197]],
            [[37.02480474373492], [54.750392033622], [5.8903503351407185], [19.910444601774977],
             [16.854308862139277],
             [4.056668891359296], [37.02480474373492], [54.750392033622], [5.8903503351407185],
             [19.910444601774977],
             [16.854308862139277], [4.056668891359296], [37.02480474373492], [54.750392033622],
             [5.8903503351407185],
             [19.910444601774977], [16.854308862139277], [4.056668891359296], [37.02480474373492],
             [54.750392033622],
             [5.8903503351407185], [19.910444601774977], [16.854308862139277], [4.056668891359296],
             [37.02480474373492], [54.750392033622]],
            [[36.901430624102545], [52.90779489234828], [7.9637846221382995], [21.447009951313554],
             [18.687291974029808], [6.307953835768051], [36.901430624102545], [52.90779489234828],
             [7.9637846221382995], [21.447009951313554], [18.687291974029808], [6.307953835768051],
             [36.901430624102545], [52.90779489234828], [7.9637846221382995], [21.447009951313554],
             [18.687291974029808], [6.307953835768051], [36.901430624102545], [52.90779489234828],
             [7.9637846221382995], [21.447009951313554], [18.687291974029808], [6.307953835768051],
             [36.901430624102545], [52.90779489234828]],
            [[38.27829147810729], [57.35404479203078], [9.492384493649608], [19.86032276121565],
             [16.571399776056424],
             [7.519030702554074], [38.27829147810729], [57.35404479203078], [9.492384493649608],
             [19.86032276121565],
             [16.571399776056424], [7.519030702554074], [38.27829147810729], [57.35404479203078],
             [9.492384493649608],
             [19.86032276121565], [16.571399776056424], [7.519030702554074], [38.27829147810729],
             [57.35404479203078],
             [9.492384493649608], [19.86032276121565], [16.571399776056424], [7.519030702554074],
             [38.27829147810729],
             [57.35404479203078]]]


@pytest.fixture
def test_bootstrap_validations(test_bootstrap_models, test_bootstrap_metrics, test_bootstrap_predictions):
    validations = [
        Validation(metrics=_metric, predictions=ddf.from_pandas(pd.DataFrame(_df), npartitions=2), model=_model)
        for _metric, _df, _model in zip(test_bootstrap_metrics, test_bootstrap_predictions, test_bootstrap_models)]
    return validations


@pytest.fixture
def test_pipeline_result(test_causal_predictions, test_boosted_predictions, test_truth, test_bootstrap_validations):
    return PipelineValidation(split_validations=[ValidationSplit(split='1970-01-01 00:00:07', boosted_validations=[
        BoostValidation(metrics={'mse': 147.27655145967876}, horizon=1, predictions=test_boosted_predictions,
                        model=EWMA(alpha=0.08))],
                                                                 causal_validation=CausalValidation(
                                                                     metrics={'mse': 199.71788056125322},
                                                                     predictions=test_causal_predictions,
                                                                     bootstrap_validations=test_bootstrap_validations),

                                                                 truth=test_truth)])


@pytest.fixture
def test_kind_cluster(kind_cluster):
    kfp_version = '1.8.5'
    kind_cluster.kubectl("apply", "-k",
                         "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref={}".format(
                             kfp_version))
    kind_cluster.kubectl("wait", "--for", "condition=established", "--timeout=60s", "crd/applications.app.k8s.io")
    kind_cluster.kubectl("apply", "-k",
                         "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref={}".format(
                             kfp_version))
    for p in Pod.objects(kind_cluster.api, namespace="kubeflow").filter(selector=""):
        if p.metadata['name'].startswith('test-pipeline'):
            p.delete()
    while not all([p.obj['status']['phase'] == 'Running' for p in
                   Pod.objects(kind_cluster.api, namespace="kubeflow").filter(selector="")]):
        time.sleep(5)

    return kind_cluster


@pytest.fixture
def minio_endpoint(test_kind_cluster):
    for p in Pod.objects(test_kind_cluster.api, namespace="kubeflow").filter(selector=""):
        if p.metadata['name'].startswith('minio'):
            return 'http://{}:9000'.format(p.obj['status']['podIP'])


@pytest.fixture()
def test_bucket():
    return 'test-bucket'


@pytest.fixture
def test_boost_model_params():
    return {'alpha': 0.08}
