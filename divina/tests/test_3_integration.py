from divina.divina.pipeline.utils import get_dask_client, DaskConfiguration

from distributed import Client


def test_dask_client_aws():
    @get_dask_client
    def test(dask_configuration):
        assert Client.current == 1

    test(dask_configuration=DaskConfiguration(destination='aws', num_workers=2))
