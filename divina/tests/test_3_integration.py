from divina.divina.pipeline.utils import get_dask_client, DaskConfiguration
import dask.dataframe as dd
import pandas as pd

from distributed import Client


def test_dask_client_aws():
    @get_dask_client
    def test(dask_configuration):
        df = dd.from_pandas(pd.DataFrame([['1', '2', '3'], ['1', '2', '3']]), npartitions=2)
        df.compute()
        assert Client.current == 1

    test(dask_configuration=DaskConfiguration(destination='aws', num_workers=2, debug=True))