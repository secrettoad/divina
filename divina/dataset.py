import dask.dataframe as dd
import os
import backoff
from botocore.exceptions import ClientError


@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def get_dataset(vision_definition):

    df = dd.read_parquet(
        "{}/{}/data/*".format(
            vision_definition["dataset_directory"], vision_definition["dataset_id"]
        )
    )
    profile = dd.read_parquet(
        "{}/{}/profile/*".format(
            vision_definition["dataset_directory"], vision_definition["dataset_id"]
        )
    )
    if "joins" in vision_definition:
        for i, join in enumerate(vision_definition["joins"]):
            join_df = dd.read_parquet(
                "{}/{}/data/*".format(join["dataset_directory"], join["dataset_id"])
            )
            join_df.columns = [
                "{}_{}".format(join["dataset_id"], c) for c in join_df.columns
            ]
            df = df.merge(
                join_df,
                how="left",
                left_on=join["join_on"][0],
                right_on=join["join_on"][1],
            )

            join_profile = dd.read_parquet(
                "{}/{}/profile/*".format(join["dataset_directory"], join["dataset_id"])
            )
            join_profile.columns = [
                "{}_{}".format(join["dataset_id"], c) for c in join_profile.columns
            ]
            profile = dd.concat([profile, join_profile], axis=1)

    return df, profile


def build_dataset_dask(
    s3_fs, read_path, write_path, dataset_name, partition_dimensions=None
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
        df.to_parquet("{}/{}/data".format(write_path, dataset_name))
        df.describe().to_parquet("{}/{}/profile".format(write_path, dataset_name))
    else:
        df.to_parquet(
            "{}/{}/data".format(write_path, dataset_name),
            partition_dimensions=partition_dimensions,
        )
        df.describe().to_parquet(
            "{}/{}/profile".format(write_path, dataset_name),
            partition_dimensions=partition_dimensions,
        )
