import os
from ..cli.cli import (
    cli_train_experiment,
    cli_forecast_experiment,
    cli_validate_experiment,
)
import dask.dataframe as ddf
import pandas as pd
import joblib
from ..utils import compare_sk_models
import json
import pytest
import pathlib
import shutil
from ..experiment import _experiment
import fsspec

@pytest.fixture(autouse=True)
def setup_teardown(setup_teardown_test_bucket_contents):
    pass


def test_train_small(
        s3_fs, test_df_1, test_model_1, test_fd_3, dask_client_remote, test_bucket, test_bootstrap_models, random_state
):
    experiment_path = "{}/experiment/test1".format(test_bucket)
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_fd_3["experiment_definition"]["data_path"]
    )
    cli_train_experiment(
        s3_fs=s3_fs,
        experiment_definition=test_fd_3["experiment_definition"],
        write_path=experiment_path,
        keep_instances_alive=False,
        dask_client=dask_client_remote,
        ec2_keypair_name="divina2",
        random_state=random_state
    )
    with s3_fs.open(
            os.path.join(
                experiment_path,
                "models",
                "h-1",
            ),
            "rb",
    ) as f:
        compare_sk_models(joblib.load(f), test_model_1[0])
    for seed in test_bootstrap_models:
        with s3_fs.open(
                os.path.join(
                    experiment_path,
                    "models/bootstrap",
                    "h-1_r-{}".format(seed),
                ),
                "rb",
        ) as f:
            compare_sk_models(joblib.load(f), test_bootstrap_models[seed][0])

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
    experiment_path = "{}/experiment/test1".format(test_bucket)
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_fd_3["experiment_definition"]["data_path"],
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
            experiment_path,
            "models"
        ),
        recursive=True,
    )
    shutil.rmtree('models', ignore_errors=True)
    cli_forecast_experiment(
        s3_fs=s3_fs,
        experiment_definition=test_fd_3["experiment_definition"],
        write_path=experiment_path,
        read_path=experiment_path,
        keep_instances_alive=False,
        dask_client=dask_client_remote
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                experiment_path,
                "forecast"
            )
        ).compute().reset_index(drop=True),
        test_forecast_1.reset_index(drop=True), check_dtype=False
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
    experiment_path = "{}/experiment/test1".format(test_bucket)
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_fd_3["experiment_definition"]["data_path"]
    )
    for split in test_validation_models:
        with s3_fs.open("{}/{}/{}".format(experiment_path,
                                          "models",
                                          "s-{}_h-1".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S"))), 'wb'
                        ) as f:
            joblib.dump(test_validation_models[split][0], f)
        with s3_fs.open("{}/{}/{}".format(experiment_path,
                                          "models",
                                          "s-{}_h-1_params.json".format(
                                              pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S"))), 'w') as f:
            json.dump(
                test_validation_models[split][1],
                f
            )
    cli_validate_experiment(
        s3_fs=s3_fs,
        experiment_definition=test_fd_3["experiment_definition"],
        write_path=experiment_path,
        read_path=experiment_path,
        ec2_keypair_name="divina2",
        keep_instances_alive=False,
        dask_client=dask_client_remote,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                experiment_path,
                "validation",
                "s-19700101-000007",
            )
        ).compute().reset_index(drop=True),
        test_val_predictions_1.reset_index(drop=True),
    )

    with s3_fs.open(
            os.path.join(experiment_path, "metrics.json"),
            "r",
    ) as f:
        metrics = json.load(f)

    assert metrics == test_metrics_1


def test_quickstart(test_fds_quickstart, random_state, dask_client_remote, test_bucket, s3_fs):
    ###Date, Customers, Promo2, Open, Competition removed on 2
    for k in test_fds_quickstart:
        experiment_path = "{}/experiment/test1/{}".format(test_bucket, k)
        fd = test_fds_quickstart[k]
        _experiment(
            experiment_definition=fd["experiment_definition"],
            read_path=experiment_path,
            write_path=experiment_path,
            random_state=11,
            s3_fs=s3_fs
        )
        result_df = ddf.read_parquet(
            os.path.join(
                experiment_path,
                "forecast"
            )
        ).compute().reset_index(drop=True)
        pd.testing.assert_frame_equal(result_df, pd.read_parquet(pathlib.Path(pathlib.Path(__file__).parent.parent.parent, 'docs_src/results/forecasts',
                               k)).reset_index(drop=True), check_exact=False, rtol=.1)


