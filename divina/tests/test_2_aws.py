import json
import os
import pathlib
import shutil

import dask.dataframe as ddf
import joblib
import pandas as pd
import pytest

from ..experiment import Experiment
from ..utils import compare_sk_models


@pytest.fixture(autouse=True)
def setup_teardown(setup_teardown_test_bucket_contents):
    pass


def test_train(
    s3_fs,
    test_df_1,
    test_model_1,
    test_ed_3,
    dask_client_remote,
    test_bucket,
    test_bootstrap_models,
    random_state,
    test_validation_models,
):
    experiment_path = "{}/experiment/test1".format(test_bucket)
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_ed_3["experiment_definition"]["data_path"]
    )
    experiment = Experiment(**test_ed_3["experiment_definition"])
    experiment.train(
        write_path=experiment_path,
        random_state=random_state,
    )

    compare_sk_models(
        experiment.models["horizons"][1]["base"][0],
        test_model_1[0],
    )
    assert experiment.models["horizons"][1]["base"][1] == test_model_1[1]["features"]
    for state in test_bootstrap_models:
        compare_sk_models(
            experiment.models["horizons"][1]["bootstrap"][state][0],
            test_bootstrap_models[state][0],
        )
        assert (
            experiment.models["horizons"][1]["bootstrap"][state][1]
            == test_bootstrap_models[state][1]["features"]
        )
    for split in test_validation_models:
        compare_sk_models(
            experiment.models["horizons"][1]["splits"][split][0],
            test_validation_models[split][0],
        )
        assert (
            experiment.models["horizons"][1]["splits"][split][1]
            == test_validation_models[split][1]["features"]
        )
    del experiment


def test_forecast(
    s3_fs,
    test_df_1,
    test_model_1,
    test_val_predictions_1,
    test_ed_3,
    dask_client_remote,
    test_bucket,
    test_forecast_1,
    test_bootstrap_models,
    random_state,
):
    experiment_path = "{}/experiment/test1".format(test_bucket)
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_ed_3["experiment_definition"]["data_path"],
    )
    pathlib.Path("models/bootstrap").mkdir(parents=True, exist_ok=True)
    joblib.dump(test_model_1[0], "models/h-1")
    with open(
        os.path.join(
            "models",
            "h-1_params.json",
        ),
        "w+",
    ) as f:
        json.dump(test_model_1[1], f)
    for state in test_bootstrap_models:
        joblib.dump(
            test_bootstrap_models[state][0],
            os.path.join(
                "models/bootstrap",
                "h-1_r-{}".format(state),
            ),
        )
        with open(
            os.path.join(
                "models/bootstrap",
                "h-1_r-{}_params.json".format(state),
            ),
            "w+",
        ) as f:
            json.dump(test_bootstrap_models[state][1], f)
    s3_fs.put(
        "models",
        os.path.join(experiment_path, "models"),
        recursive=True,
    )
    shutil.rmtree("models", ignore_errors=True)
    experiment = Experiment(**test_ed_3["experiment_definition"])
    result = experiment.forecast(
        read_path=experiment_path,
        write_path=experiment_path,
    )
    pd.testing.assert_frame_equal(
        result.compute().reset_index(drop=True),
        test_forecast_1.reset_index(drop=True),
        check_dtype=False,
    )
    del experiment


def test_validate(
    s3_fs,
    test_ed_3,
    test_df_1,
    test_metrics_1,
    test_val_predictions_1,
    dask_client_remote,
    test_bucket,
    test_validation_models,
    test_model_1,
):
    experiment_path = "{}/experiment/test1".format(test_bucket)
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_ed_3["experiment_definition"]["data_path"]
    )
    for split in test_validation_models:
        with s3_fs.open(
            "{}/{}/{}".format(
                experiment_path,
                "models",
                "s-{}_h-1".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")),
            ),
            "wb",
        ) as f:
            joblib.dump(test_validation_models[split][0], f)
        with s3_fs.open(
            "{}/{}/{}".format(
                experiment_path,
                "models",
                "s-{}_h-1_params.json".format(
                    pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")
                ),
            ),
            "w",
        ) as f:
            json.dump(test_validation_models[split][1], f)
    experiment = Experiment(**test_ed_3["experiment_definition"])
    experiment.validate(read_path=experiment_path, write_path=experiment_path)
    assert experiment.metrics == test_metrics_1
    del experiment


def test_quickstart(
    test_eds_quickstart, random_state, dask_client_remote, test_bucket, s3_fs
):
    for k in test_eds_quickstart:
        experiment_path = "{}/experiment/test1/{}".format(test_bucket, k)
        ed = test_eds_quickstart[k]
        experiment = Experiment(**ed["experiment_definition"])
        result = experiment.run(write_path=experiment_path, random_state=11)
        result_df = result.compute().reset_index(drop=True)
        pd.testing.assert_frame_equal(
            result_df,
            pd.read_parquet(
                pathlib.Path(
                    pathlib.Path(__file__).parent.parent.parent,
                    "docs_src/results/forecasts",
                    k,
                )
            ).reset_index(drop=True),
            check_exact=False,
            rtol=0.1,
        )
        del result
