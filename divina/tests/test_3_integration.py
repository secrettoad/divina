from divina.divina.pipeline.utils import get_dask_client, DaskConfiguration
import dask.dataframe as dd
import pandas as pd
from dask_cloudprovider.aws import EC2Cluster
from pandas.testing import assert_frame_equal

from distributed import Client


def test_dask_client_aws():
    @get_dask_client
    def test(dask_configuration):
        assert type(Client.current().cluster) == EC2Cluster
        df = pd.DataFrame([['1', '2', '3'], ['1', '2', '3']])
        ddf = dd.from_pandas(df, npartitions=2)
        assert_frame_equal(ddf.compute(), df)

    test(dask_configuration=DaskConfiguration(destination='aws', num_workers=2))