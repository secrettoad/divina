import pathlib
import dask.dataframe as dd


def _load(path):
    if not path.startswith("divina//:"):
        raise Exception("Path must begin with 'divina://'")
    else:
        local_path = pathlib.Path(str(pathlib.Path(__file__).parent), 'datasets', path[9:])
    return dd.read_parquet(local_path)
