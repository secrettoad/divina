import dask.dataframe as ddf
import numpy as np
import pandas as pd
import pytest
from pandas import Timestamp

from divina.docs_src._static.pipeline_definitions import quickstart_pipelines
from divina.divina.pipeline.model import EWMA, GLM
from divina.divina.pipeline.pipeline import (
    BoostPrediction,
    BoostValidation,
    CausalPrediction,
    CausalValidation,
    Pipeline,
    PipelineFitResult,
    PipelinePredictResult,
    Validation,
    ValidationSplit,
)


@pytest.fixture()
def random_state():
    return 11


@pytest.fixture()
def test_model_1(test_df_1, random_state, test_pipeline_1):
    params = [
        0.14068077961512673,
        0.44361057156250644,
        0.44361057156250644,
        -0.0384972575277309,
        -0.06090376985793098,
        0.13594649800145278,
    ]
    intercept = 1.1195265801634458
    fit_indices = [0, 1, 2, 3, 4, 5]

    model = GLM(link_function="log")
    model.linear_model.fit(
        ddf.from_pandas(
            pd.DataFrame([np.array(params) + c for c in range(0, 2)]),
            npartitions=1,
        ).to_dask_array(lengths=True),
        ddf.from_pandas(pd.Series([intercept, intercept]), npartitions=1).to_dask_array(
            lengths=True
        ),
    )
    model.linear_model.coef_ = np.array(params)
    model.linear_model.intercept_ = intercept
    model.linear_model._coef = np.array(params + [intercept])
    model.fit_indices = fit_indices
    model.y_min = 6.0
    return model


@pytest.fixture()
def test_params_1(test_model_1):
    return test_model_1[1]


@pytest.fixture()
def test_bootstrap_models(random_state, test_pipeline_1):
    params = [
        [0.03414853476453612, -0.10244309552961739, 0.10244560429361024],
        [0.03414853476453612, -0.10244309552961739, 0.10244560429361024],
        [0.03414853476453659, -0.1024430955296182, 0.10244560429360838],
        [0.06829723895966631, -0.2048835963524191, 0.20489171687899613],
        [
            4.808703607130793e-10,
            1.0326867920301728e-09,
            1.4426129338929902e-09,
        ],
    ]
    intercepts = [
        2.1959847684472407,
        2.1959847684472407,
        2.1959847684472407,
        2.2301283465997805,
        2.1813520174870353,
    ]
    indices = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
    states = range(
        random_state,
        random_state + test_pipeline_1.bootstrap_sample,
    )
    bootstrap_models = []

    for j, i, p, f, state in zip(
        range(0, len(states)), intercepts, params, indices, states
    ):
        model = GLM(link_function="log")
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
        model.y_min = 6.0
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
    return {"mse": 205.81161535349784}


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
    df = pd.DataFrame(
        [
            [Timestamp("1970-01-01 00:00:01"), 7.779226745902227],
            [Timestamp("1970-01-01 00:00:04"), 20.513140003145686],
            [Timestamp("1970-01-01 00:00:05"), 7.143396709186792],
            [Timestamp("1970-01-01 00:00:06"), 7.143396709186792],
            [Timestamp("1970-01-01 00:00:07"), 31.9691751340238],
            [Timestamp("1970-01-01 00:00:10"), 59.4976299657191],
        ]
    )
    df.columns = ["a", "y_hat"]
    df = df.set_index("a")
    return ddf.from_pandas(df, npartitions=2)["y_hat"]


@pytest.fixture()
def test_pipeline_name():
    return "test-pipeline"


@pytest.fixture()
def test_pipeline_root(test_pipeline_name, test_bucket):
    return "s3://{}/{}".format(test_bucket, test_pipeline_name)


@pytest.fixture()
def test_pipeline_1():
    return Pipeline(
        time_index="a",
        target="c",
        validation_splits=["1970-01-01 00:00:05"],
        frequency="S",
        confidence_intervals=[90],
        bootstrap_sample=5,
        bin_features={"b": [5, 10, 15]},
        time_horizons=[1],
        boost_window=5,
    )


@pytest.fixture()
def test_simulate_end():
    return "1970-01-01 00:00:14"


@pytest.fixture()
def test_simulate_start():
    return "1970-01-01 00:00:11"


@pytest.fixture()
def test_pipeline_2(test_pipeline_root, test_horizons):
    return Pipeline(
        time_index="a",
        target="c",
        validation_splits=["1970-01-01 00:00:05"],
        frequency="S",
        confidence_intervals=[90],
        bootstrap_sample=5,
        random_seed=11,
        bin_features={"b": [5, 10, 15]},
        time_horizons=test_horizons,
        pipeline_root=test_pipeline_root,
        target_dimensions=["d", "e"],
        boost_model_params={"alpha": 0.8},
        boost_window=5,
        causal_model_params=[
            {"link_function": "log"},
            {"link_function": None},
        ],
    )


@pytest.fixture()
def test_scenarios():
    return [{"b": x} for x in [0, 1]]


@pytest.fixture()
def test_pipelines_quickstart():
    return quickstart_pipelines


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
        [
            [Timestamp("1970-01-01 00:00:04"), 5.0, 6.0, 2, 2],
            [Timestamp("1970-01-01 00:00:01"), 2.0, 3.0, 2, 2],
            [Timestamp("1970-01-01 00:00:06"), 5.0, 6.0, 1, 1],
            [Timestamp("1970-01-01 00:00:04"), 5.0, 6.0, 2, 2],
            [Timestamp("1970-01-01 00:00:10"), 11.0, 12.0, 2, 2],
            [Timestamp("1970-01-01 00:00:07"), 8.0, 9.0, 2, 2],
            [Timestamp("1970-01-01 00:00:04"), 5.0, 6.0, 2, 2],
            [Timestamp("1970-01-01 00:00:05"), 5.0, 6.0, 1, 1],
            [Timestamp("1970-01-01 00:00:01"), 2.0, 3.0, 2, 2],
            [Timestamp("1970-01-01 00:00:10"), 11.0, 12.0, 2, 2],
            [Timestamp("1970-01-01 00:00:07"), 8.0, 9.0, 1, 1],
            [Timestamp("1970-01-01 00:00:01"), 2.0, 3.0, 1, 1],
            [Timestamp("1970-01-01 00:00:10"), 11.0, 12.0, 2, 2],
            [Timestamp("1970-01-01 00:00:01"), 2.0, 3.0, 1, 1],
            [Timestamp("1970-01-01 00:00:10"), 11.0, 12.0, 1, 1],
            [Timestamp("1970-01-01 00:00:07"), 8.0, 9.0, 2, 2],
            [Timestamp("1970-01-01 00:00:10"), 11.0, 12.0, 1, 1],
            [Timestamp("1970-01-01 00:00:07"), 8.0, 9.0, 2, 2],
            [Timestamp("1970-01-01 00:00:05"), 5.0, 6.0, 1, 1],
            [Timestamp("1970-01-01 00:00:07"), 8.0, 9.0, 2, 2],
            [Timestamp("1970-01-01 00:00:04"), 5.0, 6.0, 1, 1],
            [Timestamp("1970-01-01 00:00:05"), 5.0, 6.0, 1, 1],
            [Timestamp("1970-01-01 00:00:10"), 11.0, 12.0, 1, 1],
            [Timestamp("1970-01-01 00:00:10"), 11.0, 12.0, 2, 2],
            [Timestamp("1970-01-01 00:00:07"), 8.0, 9.0, 1, 1],
        ]
    )
    df.columns = ["a", "b", "c", "e", "d"]
    return ddf.from_pandas(df, npartitions=2)


@pytest.fixture()
def test_df_1():
    df = pd.DataFrame(
        [
            [
                Timestamp("1970-01-01 00:00:01"),
                2.0,
                1.5,
                1.5,
                12.0,
                1,
                0,
                0,
                0,
            ],
            [
                Timestamp("1970-01-01 00:00:04"),
                5.0,
                1.75,
                1.75,
                24.0,
                0,
                1,
                0,
                0,
            ],
            [
                Timestamp("1970-01-01 00:00:05"),
                5.0,
                1.0,
                1.0,
                18.0,
                0,
                1,
                0,
                0,
            ],
            [Timestamp("1970-01-01 00:00:06"), 5.0, 1.0, 1.0, 6.0, 0, 1, 0, 0],
            [
                Timestamp("1970-01-01 00:00:07"),
                8.0,
                1.6666666666666667,
                1.6666666666666667,
                54.0,
                0,
                1,
                0,
                0,
            ],
            [
                Timestamp("1970-01-01 00:00:10"),
                11.0,
                1.5714285714285714,
                1.5714285714285714,
                84.0,
                0,
                0,
                1,
                0,
            ],
        ]
    )
    df.columns = [
        "a",
        "b",
        "e",
        "d",
        "c",
        "b_(-inf, 5]",
        "b_(5, 10]",
        "b_(10, 15]",
        "b_(15, inf]",
    ]
    return ddf.from_pandas(df, npartitions=2).set_index("a")


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
        [
            [Timestamp("1970-01-01 00:00:01"), 8.0, 12.0, 1, 0],
            [Timestamp("1970-01-01 00:00:04"), 20.0, 24.0, 0, 1],
            [Timestamp("1970-01-01 00:00:05"), 15.0, 18.0, 0, 1],
            [Timestamp("1970-01-01 00:00:06"), 5.0, 6.0, 1, 0],
            [Timestamp("1970-01-01 00:00:07"), 48.0, 54.0, 0, 1],
            [Timestamp("1970-01-01 00:00:10"), 77.0, 84.0, 0, 1],
            [Timestamp("1970-01-02 00:00:01"), 8.0, 12.0, 1, 0],
            [Timestamp("1970-01-02 00:00:04"), 20.0, 24.0, 0, 1],
            [Timestamp("1970-01-02 00:00:05"), 15.0, 18.0, 0, 1],
            [Timestamp("1970-01-02 00:00:06"), 5.0, 6.0, 1, 0],
            [Timestamp("1970-01-02 00:00:07"), 48.0, 54.0, 0, 1],
            [Timestamp("1970-01-02 00:00:10"), 77.0, 84.0, 0, 1],
            [Timestamp("1970-01-03 00:00:01"), 8.0, 12.0, 1, 0],
            [Timestamp("1970-01-03 00:00:04"), 20.0, 24.0, 0, 1],
            [Timestamp("1970-01-03 00:00:05"), 15.0, 18.0, 0, 1],
            [Timestamp("1970-01-03 00:00:06"), 5.0, 6.0, 1, 0],
            [Timestamp("1970-01-03 00:00:07"), 48.0, 54.0, 0, 1],
            [Timestamp("1970-01-03 00:00:10"), 77.0, 84.0, 0, 1],
            [Timestamp("1970-01-04 00:00:01"), 8.0, 12.0, 1, 0],
            [Timestamp("1970-01-04 00:00:04"), 20.0, 24.0, 0, 1],
            [Timestamp("1970-01-04 00:00:05"), 15.0, 18.0, 0, 1],
            [Timestamp("1970-01-04 00:00:06"), 5.0, 6.0, 1, 0],
            [Timestamp("1970-01-04 00:00:07"), 48.0, 54.0, 0, 1],
            [Timestamp("1970-01-04 00:00:10"), 77.0, 84.0, 0, 1],
            [Timestamp("1970-01-05 00:00:01"), 8.0, 12.0, 1, 0],
            [Timestamp("1970-01-05 00:00:04"), 20.0, 24.0, 0, 1],
            [Timestamp("1970-01-05 00:00:05"), 15.0, 18.0, 0, 1],
            [Timestamp("1970-01-05 00:00:06"), 5.0, 6.0, 1, 0],
            [Timestamp("1970-01-05 00:00:07"), 48.0, 54.0, 0, 1],
            [Timestamp("1970-01-05 00:00:10"), 77.0, 84.0, 0, 1],
        ]
    )
    df.columns = ["a", "b", "c", "b_(5, 10]", "b_(15, inf]"]
    for c in ["e", "d"]:
        R = np.random.RandomState(11)
        df[c] = R.randint(1, 3, df.shape[0])
    return ddf.from_pandas(df, npartitions=2)


@pytest.fixture
def test_boosted_predictions():
    dfs = [
        [
            ["1970-01-01 00:00:05__index__1__index__1", 5.071820219189631],
            ["1970-01-01 00:00:06__index__1__index__1", 5.071820219189631],
            ["1970-01-01 00:00:07__index__1__index__1", 6.566921089891601],
            ["1970-01-01 00:00:07__index__2__index__2", 6.566921089891601],
            ["1970-01-01 00:00:10__index__1__index__1", 8.99408950139218],
            ["1970-01-01 00:00:10__index__2__index__2", 12.453536871913283],
        ],
        [
            ["1970-01-01 00:00:05__index__1__index__1", 5.071820219189631],
            ["1970-01-01 00:00:06__index__1__index__1", 5.071820219189631],
            ["1970-01-01 00:00:07__index__1__index__1", 6.566921089891601],
            ["1970-01-01 00:00:07__index__2__index__2", 6.566921089891601],
            ["1970-01-01 00:00:10__index__1__index__1", 7.269684237286091],
            ["1970-01-01 00:00:10__index__2__index__2", 6.566921089891602],
        ],
    ]
    dfs = [
        pd.DataFrame(columns=["__target_dimension_index__", "y_hat_boosted"], data=df)
        for df in dfs
    ]
    return [
        ddf.from_pandas(df.set_index("__target_dimension_index__"), npartitions=2)[
            "y_hat_boosted"
        ]
        for df in dfs
    ]


@pytest.fixture
def test_causal_predictions():
    df = pd.DataFrame(
        [
            ["1970-01-01 00:00:05__index__1__index__1", 5.071820219189631],
            ["1970-01-01 00:00:06__index__1__index__1", 5.071820219189631],
            ["1970-01-01 00:00:07__index__1__index__1", 6.566921089891601],
            ["1970-01-01 00:00:07__index__2__index__2", 6.566921089891601],
            ["1970-01-01 00:00:10__index__1__index__1", 6.566921089891602],
            ["1970-01-01 00:00:10__index__2__index__2", 6.566921089891602],
        ]
    )
    df.columns = ["__target_dimension_index__", "y_hat"]
    return ddf.from_pandas(df.set_index("__target_dimension_index__"), npartitions=2)[
        "y_hat"
    ]


@pytest.fixture
def test_truth():
    df = pd.DataFrame(
        [
            ["1970-01-01 00:00:05__index__1__index__1", 5.0, 18.0, 0, 1, 0, 0],
            ["1970-01-01 00:00:06__index__1__index__1", 5.0, 6.0, 0, 1, 0, 0],
            ["1970-01-01 00:00:07__index__1__index__1", 8.0, 18.0, 0, 1, 0, 0],
            ["1970-01-01 00:00:07__index__2__index__2", 8.0, 36.0, 0, 1, 0, 0],
            [
                "1970-01-01 00:00:10__index__1__index__1",
                11.0,
                36.0,
                0,
                0,
                1,
                0,
            ],
            [
                "1970-01-01 00:00:10__index__2__index__2",
                11.0,
                48.0,
                0,
                0,
                1,
                0,
            ],
        ]
    )
    df.columns = [
        "__target_dimension_index__",
        "b",
        "c",
        "b_(-inf, 5]",
        "b_(5, 10]",
        "b_(10, 15]",
        "b_(15, inf]",
    ]
    return ddf.from_pandas(df.set_index("__target_dimension_index__"), npartitions=2)


@pytest.fixture
def test_bootstrap_metrics():
    return [
        {"mse": pytest.approx(643.9186200767692)},
        {"mse": pytest.approx(643.9186200767692)},
        {"mse": pytest.approx(643.9186200767671)},
        {"mse": pytest.approx(407.75609758701285)},
        {"mse": pytest.approx(833.1063483669718)},
    ]


@pytest.fixture
def test_bootstrap_predictions():
    dfs = [
        [
            ["1970-01-01 00:00:05__index__1__index__1", 4.81267427988274],
            ["1970-01-01 00:00:06__index__1__index__1", 4.81267427988274],
            ["1970-01-01 00:00:07__index__1__index__1", 6.086990574533933],
            ["1970-01-01 00:00:07__index__2__index__2", 6.086990574533933],
            ["1970-01-01 00:00:10__index__1__index__1", 6.08699057453391],
            ["1970-01-01 00:00:10__index__2__index__2", 6.08699057453391],
        ],
        [
            ["1970-01-01 00:00:05__index__1__index__1", 4.81267427988274],
            ["1970-01-01 00:00:06__index__1__index__1", 4.81267427988274],
            ["1970-01-01 00:00:07__index__1__index__1", 6.086990574533933],
            ["1970-01-01 00:00:07__index__2__index__2", 6.086990574533933],
            ["1970-01-01 00:00:10__index__1__index__1", 6.08699057453391],
            ["1970-01-01 00:00:10__index__2__index__2", 6.08699057453391],
        ],
        [
            ["1970-01-01 00:00:05__index__1__index__1", 4.812674279882751],
            ["1970-01-01 00:00:06__index__1__index__1", 4.812674279882751],
            ["1970-01-01 00:00:07__index__1__index__1", 6.086990574533964],
            ["1970-01-01 00:00:07__index__2__index__2", 6.086990574533964],
            ["1970-01-01 00:00:10__index__1__index__1", 6.086990574533981],
            ["1970-01-01 00:00:10__index__2__index__2", 6.086990574533981],
        ],
        [
            ["1970-01-01 00:00:05__index__1__index__1", 9.062803513790605],
            ["1970-01-01 00:00:06__index__1__index__1", 9.062803513790605],
            ["1970-01-01 00:00:07__index__1__index__1", 12.715358970567806],
            ["1970-01-01 00:00:07__index__2__index__2", 12.715358970567806],
            ["1970-01-01 00:00:10__index__1__index__1", 12.71535897056786],
            ["1970-01-01 00:00:10__index__2__index__2", 12.71535897056786],
        ],
        [
            ["1970-01-01 00:00:05__index__1__index__1", 1.8582747425093178],
            ["1970-01-01 00:00:06__index__1__index__1", 1.8582747425093178],
            ["1970-01-01 00:00:07__index__1__index__1", 1.858274755288365],
            ["1970-01-01 00:00:07__index__2__index__2", 1.858274755288365],
            ["1970-01-01 00:00:10__index__1__index__1", 1.858274755288349],
            ["1970-01-01 00:00:10__index__2__index__2", 1.858274755288349],
        ],
    ]
    dfs = [
        pd.DataFrame(columns=["__target_dimension_index__", "y_hat"], data=df)
        for df in dfs
    ]
    return [
        ddf.from_pandas(df.set_index("__target_dimension_index__"), npartitions=2)[
            "y_hat"
        ]
        for df in dfs
    ]


@pytest.fixture
def test_boosted_models():
    models = [EWMA(), EWMA()]
    for m in models:
        m.window = 5
    return models


@pytest.fixture
def test_boosted_metrics():
    return [
        {"mse": pytest.approx(526.3151700698562)},
        {"mse": pytest.approx(617.8586415734015)},
    ]


@pytest.fixture
def test_bootstrap_validations(
    test_bootstrap_models, test_bootstrap_metrics, test_bootstrap_predictions
):
    validations = [
        Validation(metrics=_metric, predictions=_series, model=_model)
        for _metric, _series, _model in zip(
            test_bootstrap_metrics,
            test_bootstrap_predictions,
            test_bootstrap_models,
        )
    ]
    return validations


@pytest.fixture
def test_boosted_validations(
    test_boosted_models, test_boosted_metrics, test_boosted_predictions
):
    validations = [
        BoostValidation(
            metrics=_metric,
            predictions=_series,
            model=_model,
            horizon=_horizon,
        )
        for _metric, _series, _model, _horizon in zip(
            test_boosted_metrics,
            test_boosted_predictions,
            test_boosted_models,
            [3, 4],
        )
    ]
    return validations


@pytest.fixture
def test_causal_factors():
    df = pd.DataFrame(
        [
            [
                "1970-01-01 00:00:05__index__1__index__1",
                0.17074284373414553,
                0.0,
                0.10244570624048759,
            ],
            [
                "1970-01-01 00:00:06__index__1__index__1",
                0.17074284373414553,
                0.0,
                0.10244570624048759,
            ],
            [
                "1970-01-01 00:00:07__index__1__index__1",
                0.2731885499746328,
                0.0,
                0.10244570624048759,
            ],
            [
                "1970-01-01 00:00:07__index__2__index__2",
                0.2731885499746328,
                0.0,
                0.10244570624048759,
            ],
            [
                "1970-01-01 00:00:10__index__1__index__1",
                0.3756342562151201,
                0.0,
                0.0,
            ],
            [
                "1970-01-01 00:00:10__index__2__index__2",
                0.3756342562151201,
                0.0,
                0.0,
            ],
        ]
    )
    df.columns = [
        "__target_dimension_index__",
        "factor_b",
        "factor_b_(-inf, 5]",
        "factor_b_(5, 10]",
    ]
    return ddf.from_pandas(df, npartitions=2).set_index("__target_dimension_index__")


@pytest.fixture
def test_causal_confidence_intervals():
    df = pd.DataFrame(
        [
            ["1970-01-01 00:00:05__index__1__index__1", 7.362751820227464],
            ["1970-01-01 00:00:06__index__1__index__1", 7.362751820227464],
            ["1970-01-01 00:00:07__index__1__index__1", 10.06401161215427],
            ["1970-01-01 00:00:07__index__2__index__2", 10.06401161215427],
            ["1970-01-01 00:00:10__index__1__index__1", 13.193943171049375],
            ["1970-01-01 00:00:10__index__2__index__2", 15.950627394175989],
        ]
    )
    df.columns = ["__target_dimension_index__", "c_pred_c_90"]
    return ddf.from_pandas(df, npartitions=2).set_index("__target_dimension_index__")


@pytest.fixture
def test_boosted_residual_predictions():
    dfs = [
        [
            ["1970-01-01 00:00:05__index__1__index__1", 0.0],
            ["1970-01-01 00:00:06__index__1__index__1", 0.0],
            ["1970-01-01 00:00:07__index__1__index__1", 0.0],
            ["1970-01-01 00:00:07__index__2__index__2", 0.0],
            ["1970-01-01 00:00:10__index__1__index__1", 2.4271684115005776],
            ["1970-01-01 00:00:10__index__2__index__2", 5.88661578202168],
        ],
        [
            ["1970-01-01 00:00:05__index__1__index__1", 0.0],
            ["1970-01-01 00:00:06__index__1__index__1", 0.0],
            ["1970-01-01 00:00:07__index__1__index__1", 0.0],
            ["1970-01-01 00:00:07__index__2__index__2", 0.0],
            ["1970-01-01 00:00:10__index__1__index__1", 0.7027631473944885],
            ["1970-01-01 00:00:10__index__2__index__2", 0.0],
        ],
    ]
    return [
        ddf.from_pandas(
            pd.DataFrame(
                data=df, columns=["__target_dimension_index__", "y_hat"]
            ).set_index("__target_dimension_index__"),
            npartitions=2,
        )["y_hat"]
        for df in dfs
    ]


@pytest.fixture
def test_boosted_confidence_intervals():
    dfs = [
        [
            ["1970-01-01 00:00:05__index__1__index__1", 7.362751820227464],
            ["1970-01-01 00:00:06__index__1__index__1", 7.362751820227464],
            ["1970-01-01 00:00:07__index__1__index__1", 10.06401161215427],
            ["1970-01-01 00:00:07__index__2__index__2", 10.06401161215427],
            ["1970-01-01 00:00:10__index__1__index__1", 13.193943171049375],
            ["1970-01-01 00:00:10__index__2__index__2", 15.950627394175989],
        ],
        [
            ["1970-01-01 00:00:05__index__1__index__1", 7.362751820227464],
            ["1970-01-01 00:00:06__index__1__index__1", 7.362751820227464],
            ["1970-01-01 00:00:07__index__1__index__1", 10.06401161215427],
            ["1970-01-01 00:00:07__index__2__index__2", 10.06401161215427],
            ["1970-01-01 00:00:10__index__1__index__1", 13.193943171049375],
            ["1970-01-01 00:00:10__index__2__index__2", 15.950627394175989],
        ],
    ]
    return [
        ddf.from_pandas(
            pd.DataFrame(
                data=df, columns=["__target_dimension_index__", "c_pred_c_90"]
            ).set_index("__target_dimension_index__"),
            npartitions=2,
        )
        for df in dfs
    ]


@pytest.fixture
def test_lag_features():
    dfs = [
        [
            [
                "1970-01-01 00:00:05__index__1__index__1",
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
            ],
            [
                "1970-01-01 00:00:06__index__1__index__1",
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
            ],
            [
                "1970-01-01 00:00:07__index__1__index__1",
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
            ],
            [
                "1970-01-01 00:00:07__index__2__index__2",
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
            ],
            [
                "1970-01-01 00:00:10__index__1__index__1",
                11.4330789101084,
                0.9281797808103693,
                12.92817978081037,
                np.NaN,
                np.NaN,
            ],
            [
                "1970-01-01 00:00:10__index__2__index__2",
                29.4330789101084,
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
            ],
        ],
        [
            [
                "1970-01-01 00:00:05__index__1__index__1",
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
            ],
            [
                "1970-01-01 00:00:06__index__1__index__1",
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
            ],
            [
                "1970-01-01 00:00:07__index__1__index__1",
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
            ],
            [
                "1970-01-01 00:00:07__index__2__index__2",
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
            ],
            [
                "1970-01-01 00:00:10__index__1__index__1",
                0.9281797808103693,
                12.92817978081037,
                np.NaN,
                np.NaN,
                np.NaN,
            ],
            [
                "1970-01-01 00:00:10__index__2__index__2",
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
            ],
        ],
    ]
    columns = [
        [
            "__target_dimension_index__",
            "lag_3",
            "lag_4",
            "lag_5",
            "lag_6",
            "lag_7",
        ],
        [
            "__target_dimension_index__",
            "lag_4",
            "lag_5",
            "lag_6",
            "lag_7",
            "lag_8",
        ],
    ]
    return [
        ddf.from_pandas(
            pd.DataFrame(data=df, columns=c).set_index("__target_dimension_index__"),
            npartitions=2,
        )
        for df, c in zip(dfs, columns)
    ]


@pytest.fixture
def test_pipeline_fit_result(
    test_causal_predictions,
    test_boosted_predictions,
    test_truth,
    test_bootstrap_validations,
    test_boosted_validations,
):
    return PipelineFitResult(
        split_validations=[
            ValidationSplit(
                split="1970-01-01 00:00:05",
                boosted_validations=test_boosted_validations,
                causal_validation=CausalValidation(
                    metrics={"mse": pytest.approx(624.6711566239716)},
                    predictions=test_causal_predictions,
                    bootstrap_validations=test_bootstrap_validations,
                ),
                truth=test_truth,
            )
        ]
    )


@pytest.fixture
def test_pipeline_predict_result(
    test_horizons,
    test_causal_predictions,
    test_causal_factors,
    test_causal_confidence_intervals,
    test_boosted_predictions,
    test_truth,
    test_bootstrap_validations,
    test_boosted_residual_predictions,
    test_boosted_confidence_intervals,
    test_lag_features,
    test_boosted_validations,
):
    causal_prediction = CausalPrediction(
        predictions=test_causal_predictions,
        factors=test_causal_factors,
        confidence_intervals=test_causal_confidence_intervals,
    )
    boost_predictions = [
        BoostPrediction(
            horizon=h,
            causal_predictions=causal_prediction,
            predictions=p,
            model=v.model,
            residual_predictions=r,
            confidence_intervals=i,
            lag_features=l,
        )
        for h, p, v, r, i, l in zip(
            test_horizons,
            test_boosted_predictions,
            test_boosted_validations,
            test_boosted_residual_predictions,
            test_boosted_confidence_intervals,
            test_lag_features,
        )
    ]
    return PipelinePredictResult(
        causal_predictions=causal_prediction,
        boost_predictions=boost_predictions,
        truth=test_truth.drop(columns="c"),
    )


@pytest.fixture
def test_horizon_predictions():
    horizons = {
        3: [
            ["1970-01-01 00:00:01__index__1__index__1", 0.49458147242072314],
            ["1970-01-01 00:00:01__index__2__index__2", 0.49458147242072314],
            ["1970-01-01 00:00:04__index__1__index__1", 16.603752893757296],
            ["1970-01-01 00:00:04__index__2__index__2", 16.603752893757296],
            ["1970-01-01 00:00:05__index__1__index__1", 15.722885929344612],
            ["1970-01-01 00:00:06__index__1__index__1", 15.546712536462076],
            ["1970-01-01 00:00:07__index__1__index__1", 193.26415828933065],
            ["1970-01-01 00:00:07__index__2__index__2", 195.66415828933066],
            ["1970-01-01 00:00:10__index__1__index__1", 159.3493743741311],
            ["1970-01-01 00:00:10__index__2__index__2", 163.32870249516668],
        ],
        4: [
            ["1970-01-01 00:00:01__index__1__index__1", 0.49458147242072314],
            ["1970-01-01 00:00:01__index__2__index__2", 0.49458147242072314],
            ["1970-01-01 00:00:04__index__1__index__1", 15.502669188241441],
            ["1970-01-01 00:00:04__index__2__index__2", 15.502669188241441],
            ["1970-01-01 00:00:05__index__1__index__1", 16.603752893757296],
            ["1970-01-01 00:00:06__index__1__index__1", 15.722885929344612],
            ["1970-01-01 00:00:07__index__1__index__1", 195.19992680555546],
            ["1970-01-01 00:00:07__index__2__index__2", 195.19992680555546],
            ["1970-01-01 00:00:10__index__1__index__1", 193.27922149865094],
            ["1970-01-01 00:00:10__index__2__index__2", 195.1758621038289],
        ],
    }
    columns = {3: ["index", "y_hat_boosted"], 4: ["index", "y_hat_boosted"]}
    horizons = {
        k: ddf.from_pandas(
            pd.DataFrame(data=horizons[k], columns=columns[k]), npartitions=2
        ).set_index("index")
        for k in horizons
    }
    return horizons


@pytest.fixture()
def test_bucket():
    return "test-bucket"


@pytest.fixture
def test_boost_model_params():
    return {"alpha": 0.08}
