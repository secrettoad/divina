import s3fs
import os

from divina.divina.pipeline.pipeline import (
    assert_pipeline_fit_result_equal,
    assert_pipeline_predict_result_equal,
)


def test_pipeline_fit_prefect(
    test_data_1,
    test_pipeline_2,
    test_pipeline_fit_result,
    test_boost_model_params,
    test_bucket,
    test_pipeline_root,
    test_pipeline_name,
):
    test_pipeline_2.storage_options = {
        "client_kwargs": {
            "endpoint_url": "http://{}:{}".format(os.environ["S3_HOST"], 9000)
        }
    }
    test_data_path = "{}/test-data".format(test_pipeline_root)
    fs = s3fs.S3FileSystem(**test_pipeline_2.storage_options)
    if fs.exists(test_bucket):
        fs.rm(test_bucket, True)
        fs.mkdir(test_bucket)
    else:
        fs.mkdir(test_bucket)
    test_data_1.to_parquet(
        test_data_path, storage_options=test_pipeline_2.storage_options
    )
    from prefect import flow

    @flow(name=test_pipeline_name, persist_result=True)
    def run_pipeline(df: str):
        return test_pipeline_2.fit(df=df, prefect=True)

    result = run_pipeline(test_data_path)
    assert_pipeline_fit_result_equal(result, test_pipeline_fit_result)


def test_pipeline_predict_prefect(
    test_data_1,
    test_pipeline_2,
    test_pipeline_predict_result,
    test_boost_model_params,
    test_bucket,
    test_pipeline_root,
    test_pipeline_name,
    test_bootstrap_models,
    test_boost_models,
    test_horizons,
    test_simulate_end,
    test_simulate_start,
    test_horizon_predictions,
):
    test_pipeline_2.storage_options = {
        "client_kwargs": {
            "endpoint_url": "http://{}:{}".format(os.environ["S3_HOST"], 9000)
        }
    }
    test_pipeline_2.is_fit = True
    test_pipeline_2.bootstrap_models = test_bootstrap_models
    test_pipeline_2.boost_models = test_boost_models
    fs = s3fs.S3FileSystem(**test_pipeline_2.storage_options)
    if fs.exists(test_bucket):
        fs.rm(test_bucket, True)
    else:
        fs.mkdir(test_bucket)
    from prefect import flow

    @flow(name=test_pipeline_name, persist_result=True)
    def run_pipeline():
        return test_pipeline_2.predict(
            x=test_data_1[test_data_1["a"] >= "1970-01-01 00:00:05"],
            boost_y=test_pipeline_2.target,
            horizons=test_horizons,
        )

    result = run_pipeline()
    assert_pipeline_predict_result_equal(result, test_pipeline_predict_result)
