import dask.dataframe as dd
import os
import backoff
from botocore.exceptions import ClientError


@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def get_dataset(forecast_definition):

    df = dd.read_parquet(
        "{}/data/*".format(
            forecast_definition["dataset_directory"]
        )
    )
    if "joins" in forecast_definition:
        for i, join in enumerate(forecast_definition["joins"]):
            join_df = dd.read_parquet(
                "{}/data/*".format(join["dataset_directory"])
            )
            df = df.merge(
                join_df,
                how="left",
                left_on=join["join_on"][0],
                right_on=join["join_on"][1],
                suffixes=("", "{}_".format(join['as']))
            )
    return df


def build_dataset_dask(
    s3_fs, read_path, write_path, partition_dimensions=None
):
    if write_path[:5] == "s3://":
        if not s3_fs.exists(write_path):
            s3_fs.mkdir(
                write_path,
                create_parents=True,
                region_name=os.environ["AWS_DEFAULT_REGION"],
                acl="private",
            )
    try:
        df = dd.read_parquet(read_path)
    except:
        try:
            df = dd.read_csv("{}/*.csv".format(read_path))
        except:
            try:
                df = dd.read_json("{}/*.json".format(read_path))
            except:
                raise Exception("Could not parse data at path: {}".format(read_path))
    if not partition_dimensions:
        df.to_parquet("{}/data".format(write_path))
        df.describe().to_parquet("{}/profile".format(write_path))
    else:
        df.to_parquet(
            "{}/data".format(write_path),
            partition_dimensions=partition_dimensions,
        )
        df.describe().to_parquet(
            "{}/profile".format(write_path),
            partition_dimensions=partition_dimensions,
        )
