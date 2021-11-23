import pathlib
import dask.dataframe as dd
import pandas as pd


def _load(path):
    if not path.startswith("divina://"):
        raise Exception("Path must begin with 'divina://'")
    else:

        local_path = pathlib.Path(str(pathlib.Path(__file__).parent), 'datasets', path[9:])
    return dd.from_pandas(pd.read_parquet(local_path), npartitions=2)
