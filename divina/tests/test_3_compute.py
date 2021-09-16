from unittest.mock import patch
import os
from ..cli.cli import (
    cli_build_dataset,
    cli_train_vision,
    cli_predict_vision,
    cli_validate_vision,
)
import dask.dataframe as ddf
import pandas as pd
import joblib
from ..utils import compare_sk_models
import json
import pytest


@pytest.fixture(autouse=True)
def setup_teardown(setup_teardown_test_bucket_contents):
    pass


def test_build_dataset_small(s3_fs, test_df_1, dask_client_remote, test_bucket):
    dataset_path = "{}/dataset/test1".format(test_bucket)
    data_path = "{}/data".format(test_bucket)
    test_df_1.to_csv(os.path.join(data_path, "test_df_1.csv"), index=False)
    cli_build_dataset(
        s3_fs=s3_fs,
        write_path=dataset_path,
        read_path=data_path,
        ec2_keypair_name="divina2",
        dask_client=dask_client_remote,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(os.path.join(dataset_path, "data")).compute(),
        test_df_1,
    )


def test_train_small(
    s3_fs, test_df_1, test_model_1, test_fd_3, dask_client_remote, test_bucket
):
    vision_path = "{}/vision/test1".format(test_bucket)
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_3["forecast_definition"]["dataset_directory"],
            "data",
        )
    )
    forecast_definition = test_fd_3["forecast_definition"]
    cli_train_vision(
        s3_fs=s3_fs,
        forecast_definition=forecast_definition,
        write_path=vision_path,
        keep_instances_alive=False,
        dask_client=dask_client_remote,
        ec2_keypair_name="divina2",
    )
    with s3_fs.open(
        os.path.join(
            vision_path,
            "models",
            "s-19700101-000008_h-1",
        ),
        "rb",
    ) as f:
        assert compare_sk_models(joblib.load(f), test_model_1)


def test_predict_small(
    s3_fs,
    test_df_1,
    test_model_1,
    test_predictions_1,
    test_fd_3,
    dask_client_remote,
    test_bucket,
):
    vision_path = "{}/vision/test1".format(test_bucket)
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_3["forecast_definition"]["dataset_directory"],
            "data",
        )
    )
    joblib.dump(test_model_1, "s-19700101-000008_h-1")
    s3_fs.put(
        "s-19700101-000008_h-1",
        os.path.join(
            vision_path,
            "models",
            "s-19700101-000008_h-1",
        ),
        recursive=True,
    )
    os.remove("s-19700101-000008_h-1")
    forecast_definition = test_fd_3["forecast_definition"]
    cli_predict_vision(
        s3_fs=s3_fs,
        forecast_definition=forecast_definition,
        write_path=vision_path,
        read_path=vision_path,
        keep_instances_alive=False,
        dask_client=dask_client_remote,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                vision_path,
                "predictions",
                "s-19700101-000006",
            )
        ).compute(),
        test_predictions_1,
    )


def test_validate_small(
    s3_fs,
    test_fd_3,
    test_df_1,
    test_metrics_1,
    test_predictions_1,
    dask_client_remote,
    test_bucket,
):
    vision_path = "{}/vision/test1".format(test_bucket)
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_fd_3["forecast_definition"]["dataset_directory"],
            "data",
        )
    )

    ddf.from_pandas(test_predictions_1, chunksize=10000).to_parquet(
        os.path.join(
            vision_path,
            "predictions",
            "s-19700101-000008",
        )
    )

    cli_validate_vision(
        s3_fs=s3_fs,
        forecast_definition=test_fd_3["forecast_definition"],
        write_path=vision_path,
        read_path=vision_path,
        ec2_keypair_name="divina2",
        keep_instances_alive=False,
        local=False,
        dask_client=dask_client_remote,
    )

    with s3_fs.open(
        os.path.join(vision_path, "metrics.json"),
        "r",
    ) as f:
        metrics = json.load(f)

    assert metrics == test_metrics_1
