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
from pipeline.pipeline import Pipeline, PipelineFitResult, ValidationSplit, BoostValidation, CausalValidation, \
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
    params = [[0.7015825094585276, 2.104747528375583, 2.104747528375583],
              [0.7015825094585276, 2.104747528375583, 2.1047475283755825],
              [0.7015825094585276, 2.104747528375583, 2.104747528375583],
              [0.887439478570229, 2.662318435710687, 2.6623184357106866],
              [0.887439478570229, 2.6623184357106866, 2.6623184357106866]]
    intercepts = [-3.217307556751165, -3.217307556751165, -3.217307556751165, -3.8887296321362594, -3.8887296321362594]
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
        validation_splits=["1970-01-01 00:00:05"],
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
                    validation_splits=["1970-01-01 00:00:05"],
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
                    boost_window=5, causal_model_params={'link_function': 'log'})


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
    df = pd.DataFrame(
        [[Timestamp('1970-01-01 00:00:04'), 5.0, 6.0, 2, 2], [Timestamp('1970-01-01 00:00:01'), 2.0, 3.0, 2, 2],
         [Timestamp('1970-01-01 00:00:06'), 5.0, 6.0, 1, 1], [Timestamp('1970-01-01 00:00:04'), 5.0, 6.0, 2, 2],
         [Timestamp('1970-01-01 00:00:10'), 11.0, 12.0, 2, 2], [Timestamp('1970-01-01 00:00:07'), 8.0, 9.0, 2, 2],
         [Timestamp('1970-01-01 00:00:04'), 5.0, 6.0, 2, 2], [Timestamp('1970-01-01 00:00:05'), 5.0, 6.0, 1, 1],
         [Timestamp('1970-01-01 00:00:01'), 2.0, 3.0, 2, 2], [Timestamp('1970-01-01 00:00:10'), 11.0, 12.0, 2, 2],
         [Timestamp('1970-01-01 00:00:07'), 8.0, 9.0, 1, 1], [Timestamp('1970-01-01 00:00:01'), 2.0, 3.0, 1, 1],
         [Timestamp('1970-01-01 00:00:10'), 11.0, 12.0, 2, 2], [Timestamp('1970-01-01 00:00:01'), 2.0, 3.0, 1, 1],
         [Timestamp('1970-01-01 00:00:10'), 11.0, 12.0, 1, 1], [Timestamp('1970-01-01 00:00:07'), 8.0, 9.0, 2, 2],
         [Timestamp('1970-01-01 00:00:10'), 11.0, 12.0, 1, 1], [Timestamp('1970-01-01 00:00:07'), 8.0, 9.0, 2, 2],
         [Timestamp('1970-01-01 00:00:05'), 5.0, 6.0, 1, 1], [Timestamp('1970-01-01 00:00:07'), 8.0, 9.0, 2, 2],
         [Timestamp('1970-01-01 00:00:04'), 5.0, 6.0, 1, 1], [Timestamp('1970-01-01 00:00:05'), 5.0, 6.0, 1, 1],
         [Timestamp('1970-01-01 00:00:10'), 11.0, 12.0, 1, 1], [Timestamp('1970-01-01 00:00:10'), 11.0, 12.0, 2, 2],
         [Timestamp('1970-01-01 00:00:07'), 8.0, 9.0, 1, 1]]
    )
    df.columns = ["a", "b", "c", "e", "d"]
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
    dfs = [[['1970-01-01 00:00:05__index__1.0__index__1.0', 16.50266918824144],
            ['1970-01-01 00:00:05__index__2.0__index__2.0', 16.50266918824144],
            ['1970-01-01 00:00:06__index__1.0__index__1.0', 16.50266918824144],
            ['1970-01-01 00:00:06__index__2.0__index__2.0', 16.50266918824144],
            ['1970-01-01 00:00:07__index__1.0__index__1.0', 196.15588345733482],
            ['1970-01-01 00:00:07__index__2.0__index__2.0', 196.15588345733482],
            ['1970-01-01 00:00:08__index__1.0__index__1.0', 196.45534961968653],
            ['1970-01-01 00:00:08__index__2.0__index__2.0', 192.85534961968654],
            ['1970-01-01 00:00:09__index__1.0__index__1.0', 194.11524285215688],
            ['1970-01-01 00:00:09__index__2.0__index__2.0', 192.19524285215687],
            ['1970-01-01 00:00:10__index__1.0__index__1.0', 160.11657864483226],
            ['1970-01-01 00:00:10__index__2.0__index__2.0', 163.33257864483227]],
           [['1970-01-01 00:00:05__index__1.0__index__1.0', 16.50266918824144],
            ['1970-01-01 00:00:05__index__2.0__index__2.0', 16.50266918824144],
            ['1970-01-01 00:00:06__index__1.0__index__1.0', 16.50266918824144],
            ['1970-01-01 00:00:06__index__2.0__index__2.0', 16.50266918824144],
            ['1970-01-01 00:00:07__index__1.0__index__1.0', 196.15588345733482],
            ['1970-01-01 00:00:07__index__2.0__index__2.0', 196.15588345733482],
            ['1970-01-01 00:00:08__index__1.0__index__1.0', 196.15588345733482],
            ['1970-01-01 00:00:08__index__2.0__index__2.0', 196.15588345733482],
            ['1970-01-01 00:00:09__index__1.0__index__1.0', 196.45534961968653],
            ['1970-01-01 00:00:09__index__2.0__index__2.0', 192.85534961968654],
            ['1970-01-01 00:00:10__index__1.0__index__1.0', 194.11524285215688],
            ['1970-01-01 00:00:10__index__2.0__index__2.0', 192.19524285215687]]]
    dfs = [pd.DataFrame(columns=['__target_dimension_index__', 'y_hat_boosted'], data=df) for df in dfs]
    return [ddf.from_pandas(df.set_index('__target_dimension_index__'), npartitions=2)['y_hat_boosted'] for df in dfs]


@pytest.fixture
def test_causal_predictions():
    df = pd.DataFrame(
        [['1970-01-01 00:00:05__index__1.0__index__1.0', 16.50266918824144],
         ['1970-01-01 00:00:05__index__2.0__index__2.0', 16.50266918824144],
         ['1970-01-01 00:00:06__index__1.0__index__1.0', 16.50266918824144],
         ['1970-01-01 00:00:06__index__2.0__index__2.0', 16.50266918824144],
         ['1970-01-01 00:00:07__index__1.0__index__1.0', 196.15588345733482],
         ['1970-01-01 00:00:07__index__2.0__index__2.0', 196.15588345733482],
         ['1970-01-01 00:00:08__index__1.0__index__1.0', 196.15588345733482],
         ['1970-01-01 00:00:08__index__2.0__index__2.0', 196.15588345733482],
         ['1970-01-01 00:00:09__index__1.0__index__1.0', 196.15588345733482],
         ['1970-01-01 00:00:09__index__2.0__index__2.0', 196.15588345733482],
         ['1970-01-01 00:00:10__index__1.0__index__1.0', 196.15588345733482],
         ['1970-01-01 00:00:10__index__2.0__index__2.0', 196.15588345733482]])
    df.columns = ['__target_dimension_index__', 'mean']
    return ddf.from_pandas(df.set_index('__target_dimension_index__'), npartitions=2)['mean']


@pytest.fixture
def test_truth():
    df = pd.DataFrame([['1970-01-01 00:00:05__index__1.0__index__1.0', 5.0, 18.0, 0, 1, 0, 0],
                       ['1970-01-01 00:00:05__index__2.0__index__2.0', 5.0, 0.0, 0, 1, 0, 0],
                       ['1970-01-01 00:00:06__index__1.0__index__1.0', 5.0, 6.0, 0, 1, 0, 0],
                       ['1970-01-01 00:00:06__index__2.0__index__2.0', 5.0, 0.0, 0, 1, 0, 0],
                       ['1970-01-01 00:00:07__index__1.0__index__1.0', 8.0, 18.0, 0, 1, 0, 0],
                       ['1970-01-01 00:00:07__index__2.0__index__2.0', 8.0, 36.0, 0, 1, 0, 0],
                       ['1970-01-01 00:00:08__index__1.0__index__1.0', 8.0, 0.0, 0, 1, 0, 0],
                       ['1970-01-01 00:00:08__index__2.0__index__2.0', 8.0, 0.0, 0, 1, 0, 0],
                       ['1970-01-01 00:00:09__index__1.0__index__1.0', 8.0, 0.0, 0, 1, 0, 0],
                       ['1970-01-01 00:00:09__index__2.0__index__2.0', 8.0, 0.0, 0, 1, 0, 0],
                       ['1970-01-01 00:00:10__index__1.0__index__1.0', 11.0, 36.0, 0, 0, 1, 0],
                       ['1970-01-01 00:00:10__index__2.0__index__2.0', 11.0, 48.0, 0, 0, 1, 0]])
    df.columns = ['__target_dimension_index__', 'b', 'c', 'b_(-inf, 5]', 'b_(5, 10]', 'b_(10, 15]', 'b_(15, inf]']
    return ddf.from_pandas(df.set_index('__target_dimension_index__'), npartitions=2)


@pytest.fixture
def test_bootstrap_metrics():
    return [{'mse': 3793.776291999151}, {'mse': 3793.776291999144}, {'mse': 3793.776291999151},
            {'mse': 76580.38680485822}, {'mse': 76580.38680485822}]


@pytest.fixture
def test_bootstrap_predictions():
    dfs = [[['1970-01-01 00:00:05__index__1.0__index__1.0', 10.972065237969069],
            ['1970-01-01 00:00:05__index__2.0__index__2.0', 10.972065237969069],
            ['1970-01-01 00:00:06__index__1.0__index__1.0', 10.972065237969069],
            ['1970-01-01 00:00:06__index__2.0__index__2.0', 10.972065237969069],
            ['1970-01-01 00:00:07__index__1.0__index__1.0', 90.02613772132605],
            ['1970-01-01 00:00:07__index__2.0__index__2.0', 90.02613772132605],
            ['1970-01-01 00:00:08__index__1.0__index__1.0', 90.02613772132605],
            ['1970-01-01 00:00:08__index__2.0__index__2.0', 90.02613772132605],
            ['1970-01-01 00:00:09__index__1.0__index__1.0', 90.02613772132605],
            ['1970-01-01 00:00:09__index__2.0__index__2.0', 90.02613772132605],
            ['1970-01-01 00:00:10__index__1.0__index__1.0', 90.02613772132605],
            ['1970-01-01 00:00:10__index__2.0__index__2.0', 90.02613772132605]],
           [['1970-01-01 00:00:05__index__1.0__index__1.0', 10.97206523796906],
            ['1970-01-01 00:00:05__index__2.0__index__2.0', 10.97206523796906],
            ['1970-01-01 00:00:06__index__1.0__index__1.0', 10.97206523796906],
            ['1970-01-01 00:00:06__index__2.0__index__2.0', 10.97206523796906],
            ['1970-01-01 00:00:07__index__1.0__index__1.0', 90.02613772132597],
            ['1970-01-01 00:00:07__index__2.0__index__2.0', 90.02613772132597],
            ['1970-01-01 00:00:08__index__1.0__index__1.0', 90.02613772132597],
            ['1970-01-01 00:00:08__index__2.0__index__2.0', 90.02613772132597],
            ['1970-01-01 00:00:09__index__1.0__index__1.0', 90.02613772132597],
            ['1970-01-01 00:00:09__index__2.0__index__2.0', 90.02613772132597],
            ['1970-01-01 00:00:10__index__1.0__index__1.0', 90.02613772132605],
            ['1970-01-01 00:00:10__index__2.0__index__2.0', 90.02613772132605]],
           [['1970-01-01 00:00:05__index__1.0__index__1.0', 10.972065237969069],
            ['1970-01-01 00:00:05__index__2.0__index__2.0', 10.972065237969069],
            ['1970-01-01 00:00:06__index__1.0__index__1.0', 10.972065237969069],
            ['1970-01-01 00:00:06__index__2.0__index__2.0', 10.972065237969069],
            ['1970-01-01 00:00:07__index__1.0__index__1.0', 90.02613772132605],
            ['1970-01-01 00:00:07__index__2.0__index__2.0', 90.02613772132605],
            ['1970-01-01 00:00:08__index__1.0__index__1.0', 90.02613772132605],
            ['1970-01-01 00:00:08__index__2.0__index__2.0', 90.02613772132605],
            ['1970-01-01 00:00:09__index__1.0__index__1.0', 90.02613772132605],
            ['1970-01-01 00:00:09__index__2.0__index__2.0', 90.02613772132605],
            ['1970-01-01 00:00:10__index__1.0__index__1.0', 90.02613772132605],
            ['1970-01-01 00:00:10__index__2.0__index__2.0', 90.02613772132605]],
           [['1970-01-01 00:00:05__index__1.0__index__1.0', 24.798575113650006],
            ['1970-01-01 00:00:05__index__2.0__index__2.0', 24.798575113650006],
            ['1970-01-01 00:00:06__index__1.0__index__1.0', 24.798575113650006],
            ['1970-01-01 00:00:06__index__2.0__index__2.0', 24.798575113650006],
            ['1970-01-01 00:00:07__index__1.0__index__1.0', 355.350502061348],
            ['1970-01-01 00:00:07__index__2.0__index__2.0', 355.350502061348],
            ['1970-01-01 00:00:08__index__1.0__index__1.0', 355.350502061348],
            ['1970-01-01 00:00:08__index__2.0__index__2.0', 355.350502061348],
            ['1970-01-01 00:00:09__index__1.0__index__1.0', 355.350502061348],
            ['1970-01-01 00:00:09__index__2.0__index__2.0', 355.350502061348],
            ['1970-01-01 00:00:10__index__1.0__index__1.0', 355.350502061348],
            ['1970-01-01 00:00:10__index__2.0__index__2.0', 355.350502061348]],
           [['1970-01-01 00:00:05__index__1.0__index__1.0', 24.798575113650006],
            ['1970-01-01 00:00:05__index__2.0__index__2.0', 24.798575113650006],
            ['1970-01-01 00:00:06__index__1.0__index__1.0', 24.798575113650006],
            ['1970-01-01 00:00:06__index__2.0__index__2.0', 24.798575113650006],
            ['1970-01-01 00:00:07__index__1.0__index__1.0', 355.350502061348],
            ['1970-01-01 00:00:07__index__2.0__index__2.0', 355.350502061348],
            ['1970-01-01 00:00:08__index__1.0__index__1.0', 355.350502061348],
            ['1970-01-01 00:00:08__index__2.0__index__2.0', 355.350502061348],
            ['1970-01-01 00:00:09__index__1.0__index__1.0', 355.350502061348],
            ['1970-01-01 00:00:09__index__2.0__index__2.0', 355.350502061348],
            ['1970-01-01 00:00:10__index__1.0__index__1.0', 355.350502061348],
            ['1970-01-01 00:00:10__index__2.0__index__2.0', 355.350502061348]]]
    dfs = [pd.DataFrame(columns=['__target_dimension_index__', 'y_hat'], data=df) for df in dfs]
    return [ddf.from_pandas(df.set_index('__target_dimension_index__'), npartitions=2)['y_hat'] for df in dfs]


@pytest.fixture
def test_boosted_models():
    models = [EWMA(), EWMA()]
    for m in models:
        m.window = 5
    return models


@pytest.fixture
def test_boosted_metrics():
    return [{'mse': 19763.400666817906}, {'mse': 21381.79163676186}]


@pytest.fixture
def test_bootstrap_validations(test_bootstrap_models, test_bootstrap_metrics, test_bootstrap_predictions):
    validations = [
        Validation(metrics=_metric, predictions=_series, model=_model)
        for _metric, _series, _model in zip(test_bootstrap_metrics, test_bootstrap_predictions, test_bootstrap_models)]
    return validations


@pytest.fixture
def test_boosted_validations(test_boosted_models, test_boosted_metrics, test_boosted_predictions):
    validations = [
        BoostValidation(metrics=_metric, predictions=_series, model=_model, horizon=_horizon)
        for _metric, _series, _model, _horizon in
        zip(test_boosted_metrics, test_boosted_predictions, test_boosted_models, [3, 4])]
    return validations


@pytest.fixture
def test_pipeline_result(test_causal_predictions, test_boosted_predictions, test_truth, test_bootstrap_validations,
                         test_boosted_validations):
    return PipelineFitResult(
        split_validations=[ValidationSplit(split='1970-01-01 00:00:05', boosted_validations=test_boosted_validations,
                                           causal_validation=CausalValidation(
                                               metrics={'mse': 21629.60377712552},
                                               predictions=test_causal_predictions,
                                               bootstrap_validations=test_bootstrap_validations),

                                           truth=test_truth)])


@pytest.fixture
def test_scenario_predictions():
    scenarios = [{3: [
        [20, -196.15588345733482, -178.15588345733482, -10.502669188241441, 1.4973308117585589, -10.502669188241441,
         -46.127831315086624, '1970-01-01 00:00:11', '1.0', '1.0'],
        [21, np.nan, np.nan, np.nan, np.nan, np.nan, 0.31456719302108505, '1970-01-01 00:00:11', '1.0', '2.0'],
        [22, np.nan, np.nan, np.nan, np.nan, np.nan, 0.31456719302108505, '1970-01-01 00:00:11', '2.0', '1.0'],
        [23, -196.15588345733482, -160.15588345733482, -16.50266918824144, -16.50266918824144, 1.4973308117585589,
         -45.48079131508663, '1970-01-01 00:00:11', '2.0', '2.0'],
        [24, -196.15588345733482, -196.15588345733482, -178.15588345733482, -10.502669188241441, 1.4973308117585589,
         -48.204417029239366, '1970-01-01 00:00:12', '1.0', '1.0'],
        [25, np.nan, np.nan, np.nan, np.nan, np.nan, 0.31456719302108505, '1970-01-01 00:00:12', '1.0', '2.0'],
        [26, np.nan, np.nan, np.nan, np.nan, np.nan, 0.31456719302108505, '1970-01-01 00:00:12', '2.0', '1.0'],
        [27, -196.15588345733482, -196.15588345733482, -160.15588345733482, -16.50266918824144, -16.50266918824144,
         -48.075777029239376, '1970-01-01 00:00:12', '2.0', '2.0'],
        [28, -160.15588345733482, -196.15588345733482, -196.15588345733482, -178.15588345733482, -10.502669188241441,
         -41.420502172069924, '1970-01-01 00:00:13', '1.0', '1.0'],
        [29, np.nan, np.nan, np.nan, np.nan, np.nan, 0.31456719302108505, '1970-01-01 00:00:13', '1.0', '2.0'],
        [30, np.nan, np.nan, np.nan, np.nan, np.nan, 0.31456719302108505, '1970-01-01 00:00:13', '2.0', '1.0'],
        [31, -148.15588345733482, -196.15588345733482, -196.15588345733482, -160.15588345733482, -16.50266918824144,
         -38.99362217206992, '1970-01-01 00:00:13', '2.0', '2.0'],
        [32, -0.31456719302108505, -160.15588345733482, -196.15588345733482, -196.15588345733482, -178.15588345733482,
         -8.094687947773284, '1970-01-01 00:00:14', '1.0', '1.0'],
        [33, -0.31456719302108505, np.nan, np.nan, np.nan, np.nan, 0.25165375441686805, '1970-01-01 00:00:14', '1.0',
         '2.0'],
        [34, -0.31456719302108505, np.nan, np.nan, np.nan, np.nan, 0.25165375441686805, '1970-01-01 00:00:14', '2.0',
         '1.0'],
        [35, -0.31456719302108505, -148.15588345733482, -196.15588345733482, -196.15588345733482, -160.15588345733482,
         -7.608927947773283, '1970-01-01 00:00:14', '2.0', '2.0']], 4: [
        [20, -178.15588345733482, -10.502669188241441, 1.4973308117585589, -10.502669188241441, -1.4945814724207231,
         -35.74202015625384, '1970-01-01 00:00:11', '1.0', '1.0'],
        [21, np.nan, np.nan, np.nan, np.nan, np.nan, 0.31456719302108505, '1970-01-01 00:00:11', '1.0', '2.0'],
        [22, np.nan, np.nan, np.nan, np.nan, np.nan, 0.31456719302108505, '1970-01-01 00:00:11', '2.0', '1.0'],
        [23, -160.15588345733482, -16.50266918824144, -16.50266918824144, 1.4973308117585589, -1.4945814724207231,
         -32.50682015625383, '1970-01-01 00:00:11', '2.0', '2.0'],
        [24, -196.15588345733482, -178.15588345733482, -10.502669188241441, 1.4973308117585589, -10.502669188241441,
         -46.127831315086624, '1970-01-01 00:00:12', '1.0', '1.0'],
        [25, np.nan, np.nan, np.nan, np.nan, np.nan, 0.31456719302108505, '1970-01-01 00:00:12', '1.0', '2.0'],
        [26, np.nan, np.nan, np.nan, np.nan, np.nan, 0.31456719302108505, '1970-01-01 00:00:12', '2.0', '1.0'],
        [27, -196.15588345733482, -160.15588345733482, -16.50266918824144, -16.50266918824144, 1.4973308117585589,
         -45.48079131508663, '1970-01-01 00:00:12', '2.0', '2.0'],
        [28, -196.15588345733482, -196.15588345733482, -178.15588345733482, -10.502669188241441, 1.4973308117585589,
         -48.204417029239366, '1970-01-01 00:00:13', '1.0', '1.0'],
        [29, np.nan, np.nan, np.nan, np.nan, np.nan, 0.31456719302108505, '1970-01-01 00:00:13', '1.0', '2.0'],
        [30, np.nan, np.nan, np.nan, np.nan, np.nan, 0.31456719302108505, '1970-01-01 00:00:13', '2.0', '1.0'],
        [31, -196.15588345733482, -196.15588345733482, -160.15588345733482, -16.50266918824144, -16.50266918824144,
         -48.075777029239376, '1970-01-01 00:00:13', '2.0', '2.0'],
        [32, -160.15588345733482, -196.15588345733482, -196.15588345733482, -178.15588345733482, -10.502669188241441,
         -41.420502172069924, '1970-01-01 00:00:14', '1.0', '1.0'],
        [33, np.nan, np.nan, np.nan, np.nan, np.nan, 0.31456719302108505, '1970-01-01 00:00:14', '1.0', '2.0'],
        [34, np.nan, np.nan, np.nan, np.nan, np.nan, 0.31456719302108505, '1970-01-01 00:00:14', '2.0', '1.0'],
        [35, -148.15588345733482, -196.15588345733482, -196.15588345733482, -160.15588345733482, -16.50266918824144,
         -38.99362217206992, '1970-01-01 00:00:14', '2.0', '2.0']]}, {3: [
        [20, -196.15588345733482, -178.15588345733482, -10.502669188241441, 1.4973308117585589, -10.502669188241441,
         -45.75959636628589, '1970-01-01 00:00:11', '1.0', '1.0'],
        [21, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6828021418218136, '1970-01-01 00:00:11', '1.0', '2.0'],
        [22, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6828021418218136, '1970-01-01 00:00:11', '2.0', '1.0'],
        [23, -196.15588345733482, -160.15588345733482, -16.50266918824144, -16.50266918824144, 1.4973308117585589,
         -45.112556366285894, '1970-01-01 00:00:11', '2.0', '2.0'],
        [24, -196.15588345733482, -196.15588345733482, -178.15588345733482, -10.502669188241441, 1.4973308117585589,
         -47.83618208043863, '1970-01-01 00:00:12', '1.0', '1.0'],
        [25, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6828021418218136, '1970-01-01 00:00:12', '1.0', '2.0'],
        [26, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6828021418218136, '1970-01-01 00:00:12', '2.0', '1.0'],
        [27, -196.15588345733482, -196.15588345733482, -160.15588345733482, -16.50266918824144, -16.50266918824144,
         -47.70754208043864, '1970-01-01 00:00:12', '2.0', '2.0'],
        [28, -160.15588345733482, -196.15588345733482, -196.15588345733482, -178.15588345733482, -10.502669188241441,
         -41.05226722326919, '1970-01-01 00:00:13', '1.0', '1.0'],
        [29, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6828021418218136, '1970-01-01 00:00:13', '1.0', '2.0'],
        [30, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6828021418218136, '1970-01-01 00:00:13', '2.0', '1.0'],
        [31, -148.15588345733482, -196.15588345733482, -196.15588345733482, -160.15588345733482, -16.50266918824144,
         -38.625387223269186, '1970-01-01 00:00:13', '2.0', '2.0'],
        [32, -0.6828021418218136, -160.15588345733482, -196.15588345733482, -196.15588345733482, -178.15588345733482,
         -7.800099988732702, '1970-01-01 00:00:14', '1.0', '1.0'],
        [33, -0.6828021418218136, np.nan, np.nan, np.nan, np.nan, 0.5462417134574509, '1970-01-01 00:00:14', '1.0',
         '2.0'],
        [34, -0.6828021418218136, np.nan, np.nan, np.nan, np.nan, 0.5462417134574509, '1970-01-01 00:00:14', '2.0',
         '1.0'],
        [35, -0.6828021418218136, -148.15588345733482, -196.15588345733482, -196.15588345733482, -160.15588345733482,
         -7.3143399887327, '1970-01-01 00:00:14', '2.0', '2.0']], 4: [
        [20, -178.15588345733482, -10.502669188241441, 1.4973308117585589, -10.502669188241441, -1.4945814724207231,
         -35.3737852074531, '1970-01-01 00:00:11', '1.0', '1.0'],
        [21, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6828021418218136, '1970-01-01 00:00:11', '1.0', '2.0'],
        [22, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6828021418218136, '1970-01-01 00:00:11', '2.0', '1.0'],
        [23, -160.15588345733482, -16.50266918824144, -16.50266918824144, 1.4973308117585589, -1.4945814724207231,
         -32.1385852074531, '1970-01-01 00:00:11', '2.0', '2.0'],
        [24, -196.15588345733482, -178.15588345733482, -10.502669188241441, 1.4973308117585589, -10.502669188241441,
         -45.75959636628589, '1970-01-01 00:00:12', '1.0', '1.0'],
        [25, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6828021418218136, '1970-01-01 00:00:12', '1.0', '2.0'],
        [26, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6828021418218136, '1970-01-01 00:00:12', '2.0', '1.0'],
        [27, -196.15588345733482, -160.15588345733482, -16.50266918824144, -16.50266918824144, 1.4973308117585589,
         -45.112556366285894, '1970-01-01 00:00:12', '2.0', '2.0'],
        [28, -196.15588345733482, -196.15588345733482, -178.15588345733482, -10.502669188241441, 1.4973308117585589,
         -47.83618208043863, '1970-01-01 00:00:13', '1.0', '1.0'],
        [29, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6828021418218136, '1970-01-01 00:00:13', '1.0', '2.0'],
        [30, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6828021418218136, '1970-01-01 00:00:13', '2.0', '1.0'],
        [31, -196.15588345733482, -196.15588345733482, -160.15588345733482, -16.50266918824144, -16.50266918824144,
         -47.70754208043864, '1970-01-01 00:00:13', '2.0', '2.0'],
        [32, -160.15588345733482, -196.15588345733482, -196.15588345733482, -178.15588345733482, -10.502669188241441,
         -41.05226722326919, '1970-01-01 00:00:14', '1.0', '1.0'],
        [33, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6828021418218136, '1970-01-01 00:00:14', '1.0', '2.0'],
        [34, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6828021418218136, '1970-01-01 00:00:14', '2.0', '1.0'],
        [35, -148.15588345733482, -196.15588345733482, -196.15588345733482, -160.15588345733482, -16.50266918824144,
         -38.625387223269186, '1970-01-01 00:00:14', '2.0', '2.0']]}]
    columns = {3: ['index', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'y_hat_boosted', 'a', 'd', 'e'],
               4: ['index', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'y_hat_boosted', 'a', 'd', 'e']}
    scenarios = [{k: ddf.from_pandas(
        pd.DataFrame(data=s[k], columns=columns[k]),
        npartitions=2).set_index('index') for k in s} for s in scenarios]
    return scenarios


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
