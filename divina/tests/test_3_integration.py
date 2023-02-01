from divina.divina.pipeline.utils import get_dask_client, DaskConfiguration
from divina.divina.pipeline.pipeline import assert_pipeline_fit_result_equal


def test_dask_client_aws(
    test_data_1,
    test_pipeline_2,
    test_pipeline_fit_result,
    test_boost_model_params,
    test_bucket,
    test_pipeline_root,
    test_pipeline_name,
):
    from prefect import flow

    @flow(name=test_pipeline_name, persist_result=True)
    def test_flow():
        @get_dask_client
        def run_pipeline(dask_configuration: DaskConfiguration):
            return test_pipeline_2.fit(df=test_data_1, prefect=True)

        return run_pipeline(
            dask_configuration=DaskConfiguration(
                destination="aws", num_workers=2, docker_image="jhurdle/divina:test"
            )
        )

    test_pipeline_2.pipeline_root = None
    result = test_flow()
    assert_pipeline_fit_result_equal(result, test_pipeline_fit_result)

