import os
from ..dataset import build_dataset_dask
from dask import dataframe as ddf
import pandas as pd
import json
from ..utils import compare_sk_models
from dask_ml.linear_model import LinearRegression
from ..train import dask_train
import joblib
from ..predict import dask_predict
from ..validate import dask_validate
from ..vision import set_parameters, get_parameters
import pytest


@pytest.fixture(autouse=True)
def setup_teardown(setup_teardown_test_bucket_contents):
    pass


def test_build_dataset(s3_fs, test_df_1, test_bucket):
    dataset_path = "{}/dataset/test1".format(test_bucket)
    data_path = "{}/data".format(test_bucket)
    test_df_1.to_csv(os.path.join(data_path, "test_df_1.csv"), index=False)
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(test_bucket, "test1", "data")
    )
    build_dataset_dask(
        s3_fs=s3_fs,
        write_path=dataset_path,
        read_path=data_path,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet("{}/data".format(dataset_path)).compute(),
        ddf.read_csv("{}/test_df_1.csv".format(data_path)).compute(),
    )


def test_train(s3_fs, test_df_1, test_fd_1, dask_client, test_model_1, test_bucket, test_params_1_10, test_params_1_90):
    vision_path = "{}/vision/test1".format(test_bucket)
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_1["forecast_definition"]["dataset_directory"],
            "data",
        )
    )
    with open(
        os.path.join(
            test_fd_1["forecast_definition"]["dataset_directory"],
            "forecast_definition.json",
        ),
        "w+",
    ) as f:
        json.dump(test_fd_1, f)
    dask_train(
        s3_fs=s3_fs,
        dask_model=LinearRegression,
        forecast_definition=test_fd_1["forecast_definition"],
        write_path=vision_path,
    )

    with s3_fs.open(
        os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1",
        ),
        "rb",
    ) as f:
        assert compare_sk_models(joblib.load(f), test_model_1)
    test_model_1.coef_ = test_params_1_10['params'].values()
    with s3_fs.open(
        os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1_c-10",
        ),
        "rb",
    ) as f:
        assert compare_sk_models(joblib.load(f), test_model_1)
    test_model_1.coef_ = test_params_1_90['params'].values()
    with s3_fs.open(
        os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1_c-10",
        ),
        "rb",
    ) as f:
        assert compare_sk_models(joblib.load(f), test_model_1)


def test_predict(
    s3_fs,
    dask_client,
    test_df_1,
    test_fd_1,
    test_val_predictions_1,
    test_model_1,
    test_forecast_1,
    test_bucket,
):
    vision_path = "{}/vision/test1".format(test_bucket)
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_1["forecast_definition"]["dataset_directory"],
            "data",
        )
    )
    ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(
        os.path.join(
            test_fd_1["forecast_definition"]["dataset_directory"],
            "profile",
        )
    )
    joblib.dump(test_model_1, "s-19700101-000007_h-1")
    s3_fs.put(
        "s-19700101-000007_h-1",
        os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1",
        ),
        recursive=True,
    )
    os.remove("s-19700101-000007_h-1")

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
        ).compute().reset_index(drop=True),
        test_val_predictions_1,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                vision_path,
                "predictions",
                "s-19700101-000007_forecast",
            )
        ).compute().reset_index(drop=True),
        test_forecast_1,
    )


def test_validate(
    s3_fs,
    test_fd_1,
    test_df_1,
    test_metrics_1,
    test_val_predictions_1,
    dask_client,
    test_bucket,
):
    vision_path = "{}/vision/test1".format(test_bucket)
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_1["forecast_definition"]["dataset_directory"],
            "data",
        )
    )
    ddf.from_pandas(test_val_predictions_1, chunksize=10000).to_parquet(
        os.path.join(
            vision_path,
            "predictions",
            "s-19700101-000007",
        )
    )

    dask_validate(
        s3_fs=s3_fs,
        forecast_definition=test_fd_1["forecast_definition"],
        read_path=vision_path,
        write_path=vision_path,
    )

    with s3_fs.open(
        os.path.join(vision_path, "metrics.json"),
        "r",
    ) as f:
        metrics = json.load(f)

    assert metrics == test_metrics_1


def test_get_params(
        s3_fs, test_model_1, test_params_1, test_bucket
):
    vision_path = "{}/vision/test1".format(test_bucket)
    joblib.dump(test_model_1, "s-19700101-000007_h-1")
    s3_fs.put(
        "s-19700101-000007_h-1",
        os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1",
        ),
        recursive=True,
    )
    os.remove("s-19700101-000007_h-1")
    with s3_fs.open(os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1_params",
    ), 'w') as f:
        json.dump({"params": {feature: coef for feature, coef in zip(["b"], test_model_1._coef)}}, f)
    params = get_parameters(s3_fs=s3_fs, model_path=os.path.join(
        vision_path,
        "models",
        "s-19700101-000007_h-1",
    ))

    assert params == test_params_1


def test_set_params(
        s3_fs, test_model_1, test_params_1, test_params_2, test_bucket
):
    vision_path = "{}/vision/test1".format(test_bucket)
    joblib.dump(test_model_1, "s-19700101-000007_h-1")
    s3_fs.put(
        "s-19700101-000007_h-1",
        os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1",
        ),
        recursive=True,
    )
    os.remove("s-19700101-000007_h-1")
    with s3_fs.open(os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1_params",
    ), 'w') as f:
        json.dump(test_params_1, f)
    set_parameters(s3_fs=s3_fs, model_path=os.path.join(
        vision_path,
        "models",
        "s-19700101-000007_h-1",
    ), params=test_params_2['params'])

    with s3_fs.open(os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1_params",
    ), 'rb') as f:
        params = json.load(f)

    assert params == test_params_2
