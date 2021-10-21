import os
import json
from ..forecast import get_parameters, set_parameters
from ..train import dask_train
from ..predict import dask_predict
from ..dataset import get_dataset
from ..validate import dask_validate
import pathlib
from dask_ml.linear_model import LinearRegression
from ..utils import compare_sk_models
import joblib
import pandas as pd
import dask.dataframe as ddf
from jsonschema import validate
from jsonschema.exceptions import ValidationError


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
        try:
            with open(pathlib.Path(pathlib.Path(__file__).parent.parent, 'config/fd_schema.json'), 'r') as f:
                validate(instance=dd, schema=json.load(f))
        except ValidationError:
            assert True
        else:
            assert False


def test_dask_train(s3_fs, test_df_1, test_fd_1, test_model_1, dask_client, test_bootstrap_models, random_state):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(
        test_fd_1["forecast_definition"]["dataset_directory"],
    ).mkdir(parents=True, exist_ok=True)
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        test_fd_1["forecast_definition"]["dataset_directory"]
    )
    dask_train(
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
        test_model_1,
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
            test_bootstrap_models[seed],
        )


def test_dask_train_retail(s3_fs, test_df_retail_sales, test_df_retail_stores, test_df_retail_time, test_fd_retail,
                           test_model_retail, dask_client, test_bootstrap_models_retail, random_state):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(
        os.path.join(
            test_fd_retail["forecast_definition"]["dataset_directory"],
        )
    ).mkdir(parents=True, exist_ok=True)
    ddf.from_pandas(test_df_retail_sales, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_stores, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["joins"][1]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_time, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["joins"][0]["dataset_directory"],
        )
    )
    dask_train(
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
                    "s-20150718-000000_h-2",
                )
            )
        ),
        test_model_retail,
    )
    for seed in test_bootstrap_models_retail:
        compare_sk_models(
            joblib.load(
                os.path.abspath(
                    os.path.join(
                        vision_path,
                        "models/bootstrap",
                        "s-20150718-000000_h-2_r-{}".format(seed),
                    )
                )
            ),
            test_bootstrap_models_retail[seed],
        )


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

    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        test_fd_2["forecast_definition"]["dataset_directory"]
    )
    ddf.from_pandas(test_df_2, chunksize=10000).to_parquet(

        test_fd_2["forecast_definition"]["joins"][0]["dataset_directory"]
    )
    df = get_dataset(test_fd_2["forecast_definition"])

    pd.testing.assert_frame_equal(df.compute(), test_composite_dataset_1)


def test_dask_predict(
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
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        test_fd_1["forecast_definition"]["dataset_directory"]
    )
    joblib.dump(
        test_model_1,
        os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1",
        ),
    )
    for seed in test_bootstrap_models:
        joblib.dump(
            test_bootstrap_models[seed],
            os.path.join(
                vision_path,
                "models/bootstrap",
                "s-19700101-000007_h-1_r-{}".format(seed),
            ),
        )

    dask_predict(
        s3_fs=s3_fs,
        forecast_definition=test_fd_1["forecast_definition"],
        read_path=vision_path,
        write_path=vision_path,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                vision_path,
                "predictions",
                "s-19700101-000007",
            )
        ).compute(),
        test_val_predictions_1,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                vision_path,
                "predictions",
                "s-19700101-000007_forecast",
            )
        ).compute(),
        test_forecast_1,
    )


def test_dask_predict_retail(s3_fs, test_df_retail_sales, test_df_retail_stores, test_df_retail_time, test_fd_retail,
                             test_model_retail, test_val_predictions_retail, test_forecast_retail,
                             test_bootstrap_models_retail, dask_client, random_state):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(
        os.path.join(
            vision_path, "models", "bootstrap"
        )
    ).mkdir(parents=True, exist_ok=True)
    ddf.from_pandas(test_df_retail_sales, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_stores, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["joins"][1]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_time, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["joins"][0]["dataset_directory"],
        )
    )
    joblib.dump(
        test_model_retail,
        os.path.join(
            vision_path,
            "models",
            "s-20150718-000000_h-2",
        ),
    )
    for seed in test_bootstrap_models_retail:
        joblib.dump(
            test_bootstrap_models_retail[seed],
            os.path.join(
                vision_path,
                "models/bootstrap",
                "s-20150718-000000_h-2_r-{}".format(seed),
            ),
        )
    dask_predict(
        s3_fs=s3_fs,
        forecast_definition=test_fd_retail["forecast_definition"],
        read_path=vision_path,
        write_path=vision_path
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                vision_path,
                "predictions",
                "s-20150718-000000",
            )
        ).compute(),
        test_val_predictions_retail,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                vision_path,
                "predictions",
                "s-20150718-000000_forecast",
            )
        ).compute(),
        test_forecast_retail,
    )


def test_dask_validate(
        s3_fs, test_fd_1, test_df_1, test_metrics_1, test_val_predictions_1, dask_client
):
    vision_path = "divina-test/vision/test1"
    ddf.from_pandas(test_val_predictions_1, chunksize=10000).to_parquet(
        os.path.join(
            vision_path,
            "predictions",
            "s-19700101-000007",
        )
    )
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        test_fd_1["forecast_definition"]["dataset_directory"]
    )
    dask_validate(
        s3_fs=s3_fs,
        forecast_definition=test_fd_1["forecast_definition"],
        read_path=vision_path,
        write_path=vision_path,
    )

    with open(
            os.path.join(vision_path, "metrics.json"),
            "r",
    ) as f:
        metrics = json.load(f)

    assert metrics == test_metrics_1


def test_dask_validate_retail(s3_fs, test_df_retail_sales, test_df_retail_stores, test_df_retail_time, test_fd_retail,
                              test_val_predictions_retail, test_metrics_retail, dask_client):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(
        os.path.join(
            vision_path, "models", "bootstrap"
        )
    ).mkdir(parents=True, exist_ok=True)
    ddf.from_pandas(test_df_retail_sales, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_stores, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["joins"][1]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_time, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_retail["forecast_definition"]["joins"][0]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_val_predictions_retail, chunksize=10000).to_parquet(
        os.path.join(
            vision_path,
            "predictions",
            "s-20150718-000000",
        )
    )
    dask_validate(
        s3_fs=s3_fs,
        forecast_definition=test_fd_retail["forecast_definition"],
        read_path=vision_path,
        write_path=vision_path
    )
    with open(
            os.path.join(vision_path, "metrics.json"),
            "r",
    ) as f:
        metrics = json.load(f)

    assert metrics == test_metrics_retail


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
        json.dump({"params": {feature: coef for feature, coef in zip(["b"], test_model_1._coef)}}, f)
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
