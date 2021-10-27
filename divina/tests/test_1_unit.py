import os
import json
from ..train import _train
from ..forecast import _forecast
from ..dataset import _get_dataset
from ..validate import _validate
import pathlib
from dask_ml.linear_model import LinearRegression
from ..utils import compare_sk_models, get_parameters, set_parameters
import joblib
import pandas as pd
import dask.dataframe as ddf
from jsonschema import validate
import plotly.graph_objects as go


def test_validate_forecast_definition(
        fd_no_target,
        fd_time_horizons_not_list,
        fd_time_validation_splits_not_list,
        fd_no_time_index,
        fd_no_dataset_directory,
        fd_invalid_model,
        fd_time_horizons_range_not_tuple
):
    for dd in [
        fd_no_target,
        fd_no_time_index,
        fd_time_validation_splits_not_list,
        fd_time_horizons_not_list,
        fd_no_dataset_directory,
        fd_invalid_model,
        fd_time_horizons_range_not_tuple
    ]:
        with open(pathlib.Path(pathlib.Path(__file__).parent.parent, 'config/fd_schema.json'), 'r') as f:
            validate(instance=dd, schema=json.load(f))
        return None


def test_get_composite_dataset(
        test_df_1,
        test_df_2,
        test_fd_2,
        test_composite_dataset_1,
        dask_client,
):
    for dataset in test_fd_2["forecast_definition"]["joins"]:
        pathlib.Path(
            dataset["dataset_directory"]
        ).mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        test_fd_2["forecast_definition"]["dataset_directory"],
    ).mkdir(parents=True, exist_ok=True)

    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_fd_2["forecast_definition"]["dataset_directory"]
    )
    ddf.from_pandas(test_df_2, npartitions=2).to_parquet(

        test_fd_2["forecast_definition"]["joins"][0]["dataset_directory"]
    )
    df = _get_dataset(test_fd_2["forecast_definition"])

    pd.testing.assert_frame_equal(df.compute().reset_index(drop=True), test_composite_dataset_1.reset_index(drop=True))


def test_train(s3_fs, test_df_1, test_fd_1, test_model_1, dask_client, test_bootstrap_models, random_state):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(
        test_fd_1["forecast_definition"]["dataset_directory"],
    ).mkdir(parents=True, exist_ok=True)
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_fd_1["forecast_definition"]["dataset_directory"]
    )
    _train(
        s3_fs=s3_fs,
        dask_model=LinearRegression,
        forecast_definition=test_fd_1["forecast_definition"],
        write_path=vision_path,
        random_seed=random_state,
    )

    compare_sk_models(
        joblib.load(
            os.path.abspath(
                os.path.join(
                    vision_path,
                    "models",
                    "s-19700101-000007_h-1",
                )
            )
        ),
        test_model_1[0],
    )
    for seed in test_bootstrap_models:
        compare_sk_models(
            joblib.load(
                os.path.abspath(
                    os.path.join(
                        vision_path,
                        "models/bootstrap",
                        "s-19700101-000007_h-1_r-{}".format(seed),
                    )
                )
            ),
            test_bootstrap_models[seed][0],
        )


def test_train_retail(s3_fs, test_df_retail_sales, test_df_retail_stores, test_df_retail_time, test_fd_retail,
                      test_model_retail, dask_client, test_bootstrap_models_retail, random_state):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(
        os.path.join(
            test_fd_retail["forecast_definition"]["dataset_directory"],
        )
    ).mkdir(parents=True, exist_ok=True)
    ddf.from_pandas(test_df_retail_sales, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_stores, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["joins"][1]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_time, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["joins"][0]["dataset_directory"],
        )
    )
    _train(
        s3_fs=s3_fs,
        dask_model=LinearRegression,
        forecast_definition=test_fd_retail["forecast_definition"],
        write_path=vision_path,
        random_seed=random_state,
    )

    compare_sk_models(
        joblib.load(
            os.path.abspath(
                os.path.join(
                    vision_path,
                    "models",
                    "h-2",
                )
            )
        ),
        test_model_retail[0],
    )
    for seed in test_bootstrap_models_retail:
        compare_sk_models(
            joblib.load(
                os.path.abspath(
                    os.path.join(
                        vision_path,
                        "models/bootstrap",
                        "h-2_r-{}".format(seed),
                    )
                )
            ),
            test_bootstrap_models_retail[seed][0],
        )


def test_forecast(
        s3_fs, dask_client, test_df_1, test_fd_1, test_model_1, test_val_predictions_1, test_forecast_1,
        test_bootstrap_models
):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(
        test_fd_1["forecast_definition"]["dataset_directory"]
    ).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(vision_path, "models/bootstrap")).mkdir(
        parents=True, exist_ok=True
    )
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_fd_1["forecast_definition"]["dataset_directory"]
    )
    joblib.dump(
        test_model_1[0],
        os.path.join(
            vision_path,
            "models",
            "h-1",
        ),
    )
    with open(os.path.join(
            vision_path,
            "models",
            "h-1_params.json",
    ), 'w+') as f:
        json.dump(
            test_model_1[1],
            f
        )
    for seed in test_bootstrap_models:
        joblib.dump(
            test_bootstrap_models[seed][0],
            os.path.join(
                vision_path,
                "models/bootstrap",
                "h-1_r-{}".format(seed),
            ),
        )
        with open(os.path.join(
                vision_path,
                "models/bootstrap",
                "h-1_r-{}_params.json".format(seed),
        ), 'w+') as f:
            json.dump(
                test_bootstrap_models[seed][1],
                f
            )

    _forecast(
        s3_fs=s3_fs,
        forecast_definition=test_fd_1["forecast_definition"],
        read_path=vision_path,
        write_path=vision_path,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                vision_path,
                "forecast"
            )
        ).compute().reset_index(drop=True),
        test_forecast_1.reset_index(drop=True),
    )


def test_forecast_retail(s3_fs, test_df_retail_sales, test_df_retail_stores, test_df_retail_time, test_fd_retail,
                         test_model_retail, test_val_predictions_retail, test_forecast_retail,
                         test_bootstrap_models_retail, dask_client, random_state):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(
        os.path.join(
            vision_path, "models", "bootstrap"
        )
    ).mkdir(parents=True, exist_ok=True)
    ddf.from_pandas(test_df_retail_sales, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_stores, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["joins"][1]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_time, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["joins"][0]["dataset_directory"],
        )
    )
    joblib.dump(
        test_model_retail[0],
        os.path.join(
            vision_path,
            "models",
            "h-2",
        ),
    )
    with open(os.path.join(
            vision_path,
            "models",
            "h-2_params.json",
    ), 'w+') as f:
        json.dump(
            test_model_retail[1],
            f
        )
    for seed in test_bootstrap_models_retail:
        joblib.dump(
            test_bootstrap_models_retail[seed][0],
            os.path.join(
                vision_path,
                "models/bootstrap",
                "h-2_r-{}".format(seed),
            ),
        )
        with open(os.path.join(
                vision_path,
                "models/bootstrap",
                "h-2_r-{}_params.json".format(seed),
        ), 'w+') as f:
            json.dump(
                test_bootstrap_models_retail[seed][1],
                f
            )
    _forecast(
        s3_fs=s3_fs,
        forecast_definition=test_fd_retail["forecast_definition"],
        read_path=vision_path,
        write_path=vision_path
    )
    result_df = ddf.read_parquet(
        os.path.join(
            vision_path,
            "forecast"
        )
    ).compute().reset_index(drop=True)
    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text="A Figure Specified By A Graph Object")
        )
    )
    fig.add_trace(go.Scatter(mode="lines", x=test_df_retail_sales[test_fd_retail["forecast_definition"]['time_index']],
                             y=test_df_retail_sales[test_fd_retail["forecast_definition"]['target']]))
    for h in test_fd_retail["forecast_definition"]['time_horizons']:
        fig.add_trace(go.Scatter(mode="lines", x=result_df[test_fd_retail["forecast_definition"]['time_index']],
                                 y=result_df['{}_h_{}_pred'.format(test_fd_retail["forecast_definition"]['target'], h)].shift(h)))
        for i in test_fd_retail["forecast_definition"]['confidence_intervals']:
            fig.add_trace(go.Scatter(mode="lines", x=result_df[test_fd_retail["forecast_definition"]['time_index']],
                                     y=result_df[
                                         '{}_h_{}_pred_c_{}'.format(test_fd_retail["forecast_definition"]['target'], h, i)].shift(h)))
    fig.write_html('test_forecast_retail.html')
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                vision_path,
                "forecast"
            )
        ).compute().reset_index(drop=True),
        test_forecast_retail.reset_index(drop=True),
    )


def test_validate(
        s3_fs, test_fd_1, test_df_1, test_metrics_1, dask_client, test_val_predictions_1, test_model_1,
        test_bootstrap_models
):
    vision_path = "divina-test/vision/test1"
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_fd_1["forecast_definition"]["dataset_directory"]
    )
    pathlib.Path(
        os.path.join(
            vision_path, "models", "bootstrap"
        )
    ).mkdir(parents=True, exist_ok=True)
    joblib.dump(
        test_model_1[0],
        os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1",
        ),
    )
    with open(os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1_params.json",
    ), 'w+') as f:
        json.dump(
            test_model_1[1],
            f
        )
    for seed in test_bootstrap_models:
        joblib.dump(
            test_bootstrap_models[seed][0],
            os.path.join(
                vision_path,
                "models/bootstrap",
                "s-19700101-000007_h-1_r-{}".format(seed),
            ),
        )
        with open(os.path.join(
                vision_path,
                "models/bootstrap",
                "s-19700101-000007_h-1_r-{}_params.json".format(seed),
        ), 'w+') as f:
            json.dump(
                test_bootstrap_models[seed][1],
                f
            )
    _validate(
        s3_fs=s3_fs,
        forecast_definition=test_fd_1["forecast_definition"],
        read_path=vision_path,
        write_path=vision_path,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                vision_path,
                "validation",
                "s-19700101-000007",
            )
        ).compute().reset_index(drop=True),
        test_val_predictions_1.reset_index(drop=True),
    )
    with open(
            os.path.join(vision_path, "metrics.json"),
            "r",
    ) as f:
        assert json.load(f) == test_metrics_1


def test_validate_retail(s3_fs, test_df_retail_sales, test_df_retail_stores, test_df_retail_time, test_fd_retail,
                         test_val_predictions_retail, test_metrics_retail, test_model_retail,
                         test_bootstrap_models_retail, dask_client):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(
        os.path.join(
            vision_path, "models", "bootstrap"
        )
    ).mkdir(parents=True, exist_ok=True)
    ddf.from_pandas(test_df_retail_sales, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_stores, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["joins"][1]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_time, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["joins"][0]["dataset_directory"],
        )
    )
    joblib.dump(
        test_model_retail[0],
        os.path.join(
            vision_path,
            "models",
            "s-20150718-000000_h-2",
        ),
    )
    with open(os.path.join(
            vision_path,
            "models",
            "s-20150718-000000_h-2_params.json",
    ), 'w+') as f:
        json.dump(
            test_model_retail[1],
            f
        )
    for seed in test_bootstrap_models_retail:
        joblib.dump(
            test_bootstrap_models_retail[seed][0],
            os.path.join(
                vision_path,
                "models/bootstrap",
                "s-20150718-000000_h-2_r-{}".format(seed),
            ),
        )
        with open(os.path.join(
                vision_path,
                "models/bootstrap",
                "s-20150718-000000_h-2_r-{}_params.json".format(seed),
        ), 'w+') as f:
            json.dump(
                test_bootstrap_models_retail[seed][1],
                f
            )
    _validate(
        s3_fs=s3_fs,
        forecast_definition=test_fd_retail["forecast_definition"],
        read_path=vision_path,
        write_path=vision_path
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                vision_path,
                "validation",
                "s-20150718-000000",
            )
        ).compute().reset_index(drop=True),
        test_val_predictions_retail.reset_index(drop=True),
    )
    with open(
            os.path.join(vision_path, "metrics.json"),
            "r",
    ) as f:
        assert json.loads(f) == test_metrics_retail


def test_get_params(
        s3_fs, test_model_1, test_params_1
):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(os.path.join(vision_path, "models")).mkdir(
        parents=True, exist_ok=True
    )
    joblib.dump(
        test_model_1,
        os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1",
        ),
    )
    with open(os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1_params",
    ), 'w+') as f:
        json.dump(test_model_1[1], f)
    params = get_parameters(s3_fs=s3_fs, model_path=os.path.join(
        vision_path,
        "models",
        "s-19700101-000007_h-1",
    ))

    assert params == test_params_1


def test_set_params(
        s3_fs, test_model_1, test_params_1, test_params_2
):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(os.path.join(vision_path, "models")).mkdir(
        parents=True, exist_ok=True
    )
    joblib.dump(
        test_model_1,
        os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1",
        ),
    )
    with open(os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1_params",
    ), 'w+') as f:
        json.dump(test_params_1, f)
    set_parameters(s3_fs=s3_fs, model_path=os.path.join(
        vision_path,
        "models",
        "s-19700101-000007_h-1",
    ), params=test_params_2['params'])

    with open(os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1_params",
    ), 'rb') as f:
        params = json.load(f)

    assert params == test_params_2
