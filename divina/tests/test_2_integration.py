import os
from ..cli.cli import (
    cli_train_vision,
    cli_forecast_vision,
    cli_validate_vision,
)
import dask.dataframe as ddf
import pandas as pd
import joblib
from ..utils import compare_sk_models
import json
import pytest
import pathlib
import shutil


@pytest.fixture(autouse=True)
def setup_teardown(setup_teardown_test_bucket_contents):
    pass


def test_train_small(
        s3_fs, test_df_1, test_model_1, test_fd_3, dask_client_remote, test_bucket, test_bootstrap_models, random_state
):
    vision_path = "{}/vision/test1".format(test_bucket)
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_fd_3["forecast_definition"]["dataset_directory"]
    )
    cli_train_vision(
        s3_fs=s3_fs,
        forecast_definition=test_fd_3["forecast_definition"],
        write_path=vision_path,
        keep_instances_alive=False,
        dask_client=dask_client_remote,
        ec2_keypair_name="divina2",
        random_state=random_state
    )
    with s3_fs.open(
            os.path.join(
                vision_path,
                "models",
                "h-1",
            ),
            "rb",
    ) as f:
        compare_sk_models(joblib.load(f), test_model_1[0])
    for seed in test_bootstrap_models:
        with s3_fs.open(
                os.path.join(
                    vision_path,
                    "models/bootstrap",
                    "h-1_r-{}".format(seed),
                ),
                "rb",
        ) as f:
            compare_sk_models(joblib.load(f), test_bootstrap_models[seed][0])


def test_dask_train_retail(s3_fs, test_df_retail_sales, test_df_retail_stores, test_df_retail_time, test_fd_retail_2,
                           test_model_retail, dask_client_remote, test_bootstrap_models_retail, test_validation_models_retail, random_state,
                           test_bucket):
    vision_path = "{}/vision/test1".format(test_bucket)
    pathlib.Path(
        os.path.join(
            test_fd_retail_2["forecast_definition"]["dataset_directory"],
        )
    ).mkdir(parents=True, exist_ok=True)
    ddf.from_pandas(test_df_retail_sales, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail_2["forecast_definition"]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_stores, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail_2["forecast_definition"]["joins"][1]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_time, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail_2["forecast_definition"]["joins"][0]["dataset_directory"],
        )
    )
    cli_train_vision(
        s3_fs=s3_fs,
        forecast_definition=test_fd_retail_2["forecast_definition"],
        write_path=vision_path,
        keep_instances_alive=False,
        dask_client=dask_client_remote,
        ec2_keypair_name="divina2",
        random_state=random_state
    )

    with s3_fs.open(
            os.path.join(
                vision_path,
                "models",
                "h-2",
            ),
            "rb",
    ) as f:
        compare_sk_models(joblib.load(f), test_model_retail[0])
    for seed in test_bootstrap_models_retail:
        with s3_fs.open(
                os.path.join(
                    vision_path,
                    "models/bootstrap",
                    "h-2_r-{}".format(seed),
                ),
                "rb",
        ) as f:
            compare_sk_models(joblib.load(f), test_bootstrap_models_retail[seed][0])
    for split in test_validation_models_retail:
        with s3_fs.open(os.path.join(
                        vision_path,
                        "models",
                        "s-{}_h-2".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")),
                    ), 'rb') as f:
            compare_sk_models(
                joblib.load(f

                ),
                test_validation_models_retail[split][0],
            )
        with s3_fs.open(os.path.join(
                os.path.join(
                    vision_path,
                    "models",
                    "s-{}_h-2_params.json".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")),
                )
        )) as f:
            features = json.load(f)
        assert features == test_validation_models_retail[split][1]


def test_forecast_small(
        s3_fs,
        test_df_1,
        test_model_1,
        test_val_predictions_1,
        test_fd_3,
        dask_client_remote,
        test_bucket,
        test_forecast_1,
        test_bootstrap_models,
        random_state
):
    vision_path = "{}/vision/test1".format(test_bucket)
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_fd_3["forecast_definition"]["dataset_directory"],
    )
    pathlib.Path(
        "models/bootstrap"
    ).mkdir(parents=True, exist_ok=True)
    joblib.dump(test_model_1[0], "models/h-1")
    with open(os.path.join(
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
                "models/bootstrap",
                "h-1_r-{}".format(seed),
            ),
        )
        with open(os.path.join(
                "models/bootstrap",
                "h-1_r-{}_params.json".format(seed),
        ), 'w+') as f:
            json.dump(
                test_bootstrap_models[seed][1],
                f
            )
    s3_fs.put(
        "models",
        os.path.join(
            vision_path,
            "models"
        ),
        recursive=True,
    )
    shutil.rmtree('models', ignore_errors=True)
    cli_forecast_vision(
        s3_fs=s3_fs,
        forecast_definition=test_fd_3["forecast_definition"],
        write_path=vision_path,
        read_path=vision_path,
        keep_instances_alive=False,
        dask_client=dask_client_remote
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                vision_path,
                "forecast"
            )
        ).compute().reset_index(drop=True),
        test_forecast_1.reset_index(drop=True), check_dtype=False
    )


def test_dask_forecast_retail(s3_fs, test_df_retail_sales, test_df_retail_stores, test_df_retail_time, test_fd_retail_2,
                              test_model_retail, test_val_predictions_retail, test_forecast_retail,
                              test_bootstrap_models_retail, dask_client_remote, test_bucket):
    vision_path = "{}/vision/test1".format(test_bucket)
    ddf.from_pandas(test_df_retail_sales, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail_2["forecast_definition"]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_stores, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail_2["forecast_definition"]["joins"][1]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_time, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail_2["forecast_definition"]["joins"][0]["dataset_directory"],
        )
    )
    with s3_fs.open("{}/{}/{}".format(vision_path,
                                      "models",
                                      "h-2"),
                    'wb') as f:
        joblib.dump(test_model_retail[0], f)

    with s3_fs.open("{}/{}/{}".format(vision_path,
                                      "models",
                                      "h-2_params.json"), 'w') as f:
        json.dump(
            test_model_retail[1],
            f
        )
    for seed in test_bootstrap_models_retail:
        with s3_fs.open(os.path.join(
                vision_path,
                "models/bootstrap",
                "h-2_r-{}".format(seed),
        ), 'wb') as f:
            joblib.dump(
                test_bootstrap_models_retail[seed][0],
                f,
            )
        with s3_fs.open(os.path.join(
                vision_path,
                "models/bootstrap",
                "h-2_r-{}_params.json".format(seed),
        ), 'w') as f:
            json.dump(
                test_bootstrap_models_retail[seed][1],
                f
            )
    cli_forecast_vision(
        s3_fs=s3_fs,
        forecast_definition=test_fd_retail_2["forecast_definition"],
        write_path=vision_path,
        read_path=vision_path,
        keep_instances_alive=False,
        dask_client=dask_client_remote
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                vision_path,
                "forecast",
            )
        ).compute().reset_index(drop=True),
        test_forecast_retail.reset_index(drop=True), check_dtype=False
    )


def test_validate_small(
        s3_fs,
        test_fd_3,
        test_df_1,
        test_metrics_1,
        test_val_predictions_1,
        dask_client_remote,
        test_bucket,
        test_validation_models,
        test_model_1
):
    vision_path = "{}/vision/test1".format(test_bucket)
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_fd_3["forecast_definition"]["dataset_directory"]
    )
    for split in test_validation_models:
        with s3_fs.open("{}/{}/{}".format(vision_path,
                                          "models",
                                          "s-{}_h-1".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S"))), 'wb'
                        ) as f:
            joblib.dump(test_validation_models[split][0], f)
        with s3_fs.open("{}/{}/{}".format(vision_path,
                                          "models",
                                          "s-{}_h-1_params.json".format(
                                              pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S"))), 'w') as f:
            json.dump(
                test_validation_models[split][1],
                f
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

    with s3_fs.open(
            os.path.join(vision_path, "metrics.json"),
            "r",
    ) as f:
        metrics = json.load(f)

    assert metrics == test_metrics_1


def test_dask_validate_retail(s3_fs, test_df_retail_sales, test_df_retail_stores, test_df_retail_time, test_fd_retail_2,
                              test_val_predictions_retail, test_validation_models_retail, test_metrics_retail, dask_client_remote, test_bucket,
                              test_model_retail):
    vision_path = "{}/vision/test1".format(test_bucket)
    ddf.from_pandas(test_df_retail_sales, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail_2["forecast_definition"]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_stores, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail_2["forecast_definition"]["joins"][1]["dataset_directory"],
        )
    )
    ddf.from_pandas(test_df_retail_time, npartitions=2).to_parquet(
        os.path.join(
            test_fd_retail_2["forecast_definition"]["joins"][0]["dataset_directory"],
        )
    )
    for split in test_validation_models_retail:
        with s3_fs.open("{}/{}/{}".format(vision_path,
                                              "models",
                                              "s-{}_h-2".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S"))), 'wb'
        ) as f:
            joblib.dump(test_validation_models_retail[split][0], f)
        with s3_fs.open("{}/{}/{}".format(vision_path,
                                              "models",
                                              "s-{}_h-2_params.json".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S"))), 'w') as f:
            json.dump(
                test_validation_models_retail[split][1],
                f
            )
    cli_validate_vision(
        s3_fs=s3_fs,
        forecast_definition=test_fd_retail_2["forecast_definition"],
        write_path=vision_path,
        read_path=vision_path,
        ec2_keypair_name="divina2",
        keep_instances_alive=False,
        local=False,
        dask_client=dask_client_remote,
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
    with s3_fs.open(
            os.path.join(vision_path, "metrics.json"),
            "r",
    ) as f:
        metrics = json.load(f)

    assert metrics == test_metrics_retail