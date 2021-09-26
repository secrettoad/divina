import os
import json
from unittest.mock import patch
from ..vision import validate_forecast_definition, get_parameters, set_parameters
from ..train import dask_train
from ..predict import dask_predict
from ..dataset import get_dataset, build_dataset_dask
from ..validate import dask_validate
from ..errors import InvalidDataDefinitionException
import pathlib
from dask_ml.linear_model import LinearRegression
from ..utils import compare_sk_models
import joblib
import pandas as pd
import dask.dataframe as ddf


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
            validate_forecast_definition(dd)
        except InvalidDataDefinitionException:
            assert True
        else:
            assert False


@patch("s3fs.S3FileSystem.open", open)
@patch("s3fs.S3FileSystem.ls", os.listdir)
@patch.dict(os.environ, {"DATA_BUCKET": "divina-test/data"})
@patch.dict(os.environ, {"DATASET_PATH": "divina-test/dataset/test1"})
def test_dataset_build(s3_fs, vision_s3, test_df_1, account_number):
    pathlib.Path(os.environ["DATA_BUCKET"]).mkdir(parents=True, exist_ok=True)
    test_df_1.to_csv(
        os.path.join(os.environ["DATA_BUCKET"], "test_df_1.csv"), index=False
    )
    pathlib.Path(os.environ["DATASET_PATH"]).mkdir(parents=True, exist_ok=True)
    build_dataset_dask(
        s3_fs=s3_fs,
        read_path=os.environ["DATA_BUCKET"],
        write_path=os.environ["DATASET_PATH"],
        partition_dimensions=None,
    )

    pd.testing.assert_frame_equal(
        ddf.read_parquet(os.path.join(os.environ["DATASET_PATH"], "data/*")).compute(),
        ddf.read_csv(os.path.join(os.environ["DATA_BUCKET"], "test_df_1.csv")).compute(),
    )


@patch("s3fs.S3FileSystem.open", open)
@patch("s3fs.S3FileSystem.ls", os.listdir)
def test_dask_train(s3_fs, test_df_1, test_fd_1, test_model_1, dask_client, test_params_1_10, test_params_1_90):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(
        os.path.join(
            test_fd_1["forecast_definition"]["dataset_directory"],
            "data",
        )
    ).mkdir(parents=True, exist_ok=True)
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_1["forecast_definition"]["dataset_directory"],
            "data",
        )
    )
    dask_train(
        s3_fs=s3_fs,
        dask_model=LinearRegression,
        forecast_definition=test_fd_1["forecast_definition"],
        write_path=vision_path,
        random_seed=11,
    )

    assert compare_sk_models(
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
    test_model_1.coef_ = test_params_1_10['params'].values()
    assert compare_sk_models(
        joblib.load(
            os.path.abspath(
                os.path.join(
                    vision_path,
                    "models",
                    "s-19700101-000007_h-1_c-10",
                )
            )
        ),
        test_model_1,
    )
    test_model_1.coef_ = test_params_1_90['params'].values()
    assert compare_sk_models(
        joblib.load(
            os.path.abspath(
                os.path.join(
                    vision_path,
                    "models",
                    "s-19700101-000007_h-1_c-90",
                )
            )
        ),
        test_model_1,
    )


@patch("s3fs.S3FileSystem.open", open)
@patch("s3fs.S3FileSystem.ls", os.listdir)
def test_get_composite_dataset(
        test_df_1,
        test_df_2,
        test_fd_2,
        test_composite_dataset_1,
        dask_client,
):
    for path in ["data"]:
        for dataset in test_fd_2["forecast_definition"]["joins"]:
            pathlib.Path(
                os.path.join(
                    dataset["dataset_directory"],
                    path,
                )
            ).mkdir(parents=True, exist_ok=True)
        pathlib.Path(
            os.path.join(
                test_fd_2["forecast_definition"]["dataset_directory"],
                path,
            )
        ).mkdir(parents=True, exist_ok=True)
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_2["forecast_definition"]["dataset_directory"],
            "data",
        )
    )
    ddf.from_pandas(test_df_2, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_2["forecast_definition"]["joins"][0]["dataset_directory"],
            "data",
        )
    )
    df = get_dataset(test_fd_2["forecast_definition"])

    pd.testing.assert_frame_equal(df.compute(), test_composite_dataset_1)


@patch("s3fs.S3FileSystem.open", open)
@patch("s3fs.S3FileSystem.ls", os.listdir)
@patch.dict(os.environ, {"VISION_PATH": "divina-test/vision/test1"})
def test_dask_predict(
        s3_fs, dask_client, test_df_1, test_fd_1, test_model_1, test_val_predictions_1, test_forecast_1, test_params_1_10, test_params_1_90
):
    pathlib.Path(
        os.path.join(
            test_fd_1["forecast_definition"]["dataset_directory"],
            "data",
        )
    ).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(os.environ["VISION_PATH"], "models")).mkdir(
        parents=True, exist_ok=True
    )
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_1["forecast_definition"]["dataset_directory"],
            "data",
        )
    )
    joblib.dump(
        test_model_1,
        os.path.join(
            os.environ["VISION_PATH"],
            "models",
            "s-19700101-000007_h-1",
        ),
    )
    test_model_1.coef_ = list(test_params_1_10["params"].values())
    joblib.dump(
        test_model_1,
        os.path.join(
            os.environ["VISION_PATH"],
            "models",
            "s-19700101-000007_h-1_c-10",
        ),
    )
    test_model_1.coef_ = list(test_params_1_90["params"].values())
    joblib.dump(
        test_model_1,
        os.path.join(
            os.environ["VISION_PATH"],
            "models",
            "s-19700101-000007_h-1_c-90",
        ),
    )
    with open(
            os.path.join(
                os.environ["VISION_PATH"],
                "forecast_definition.json",
            ),
            "w+",
    ) as f:
        json.dump(test_fd_1, f)

    dask_predict(
        s3_fs=s3_fs,
        forecast_definition=test_fd_1["forecast_definition"],
        read_path=os.environ["VISION_PATH"],
        write_path=os.environ["VISION_PATH"],
    )

    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                os.environ["VISION_PATH"],
                "predictions",
                "s-19700101-000007",
            )
        ).compute().reset_index(drop=True),
        test_val_predictions_1,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                os.environ["VISION_PATH"],
                "predictions",
                "s-19700101-000007_forecast",
            )
        ).compute().reset_index(drop=True),
        test_forecast_1,
    )


@patch("s3fs.S3FileSystem.open", open)
@patch("s3fs.S3FileSystem.ls", os.listdir)
@patch.dict(os.environ, {"VISION_PATH": "divina-test/vision/test1"})
def test_dask_validate(
        s3_fs, test_fd_1, test_df_1, test_metrics_1, test_val_predictions_1, dask_client
):
    ddf.from_pandas(test_val_predictions_1, chunksize=10000).to_parquet(
        os.path.join(
            os.environ["VISION_PATH"],
            "predictions",
            "s-19700101-000007",
        )
    )
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_1["forecast_definition"]["dataset_directory"],
            "data",
        )
    )
    dask_validate(
        s3_fs=s3_fs,
        forecast_definition=test_fd_1["forecast_definition"],
        read_path=os.environ["VISION_PATH"],
        write_path=os.environ["VISION_PATH"],
    )

    with open(
            os.path.join(os.environ["VISION_PATH"], "metrics.json"),
            "r",
    ) as f:
        metrics = json.load(f)

    assert metrics == test_metrics_1


@patch("s3fs.S3FileSystem.open", open)
@patch("s3fs.S3FileSystem.ls", os.listdir)
@patch.dict(os.environ, {"VISION_PATH": "divina-test/vision/test1"})
def test_get_params(
        s3_fs, test_model_1, test_params_1
):
    pathlib.Path(os.path.join(os.environ["VISION_PATH"], "models")).mkdir(
        parents=True, exist_ok=True
    )
    joblib.dump(
        test_model_1,
        os.path.join(
            os.environ["VISION_PATH"],
            "models",
            "s-19700101-000007_h-1",
        ),
    )
    with open(os.path.join(
            os.environ["VISION_PATH"],
            "models",
            "s-19700101-000007_h-1_params",
    ), 'w+') as f:
        json.dump({"params": {feature: coef for feature, coef in zip(["b"], test_model_1._coef)}}, f)
    params = get_parameters(s3_fs=s3_fs, model_path=os.path.join(
        os.environ["VISION_PATH"],
        "models",
        "s-19700101-000007_h-1",
    ))

    assert params == test_params_1


@patch("s3fs.S3FileSystem.open", open)
@patch("s3fs.S3FileSystem.ls", os.listdir)
@patch.dict(os.environ, {"VISION_PATH": "divina-test/vision/test1"})
def test_set_params(
        s3_fs, test_model_1, test_params_1, test_params_2
):
    pathlib.Path(os.path.join(os.environ["VISION_PATH"], "models")).mkdir(
        parents=True, exist_ok=True
    )
    joblib.dump(
        test_model_1,
        os.path.join(
            os.environ["VISION_PATH"],
            "models",
            "s-19700101-000007_h-1",
        ),
    )
    with open(os.path.join(
            os.environ["VISION_PATH"],
            "models",
            "s-19700101-000007_h-1_params",
    ), 'w+') as f:
        json.dump(test_params_1, f)
    set_parameters(s3_fs=s3_fs, model_path=os.path.join(
        os.environ["VISION_PATH"],
        "models",
        "s-19700101-000007_h-1",
    ), params=test_params_2['params'])

    with open(os.path.join(
            os.environ["VISION_PATH"],
            "models",
            "s-19700101-000007_h-1_params",
    ), 'rb') as f:
        params = json.load(f)

    assert params == test_params_2
