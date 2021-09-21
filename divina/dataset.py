import dask.dataframe as dd
import os
import backoff
from botocore.exceptions import ClientError
import pandas as pd
from .utils import cull_empty_partitions


@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def get_dataset(forecast_definition):

    df = dd.read_parquet("{}/data/*".format(forecast_definition["dataset_directory"]))

    time_min, time_max = (
        pd.to_datetime(str(df[forecast_definition["time_index"]].min().compute())),
        pd.to_datetime(str(df[forecast_definition["time_index"]].max().compute())),
    )

    if "drop_features" in forecast_definition:
        df = df.drop(columns=forecast_definition["drop_features"])

    if "signal_dimensions" in forecast_definition:
        df = df.groupby([forecast_definition["time_index"]] + forecast_definition["signal_dimensions"]).agg(
            "sum").reset_index()
    else:
        df = df.groupby(forecast_definition["time_index"]).agg("sum").reset_index()

    if "forecast_start" in forecast_definition:
        if pd.to_datetime(str(forecast_definition["forecast_start"])) < time_min:
            raise Exception(
                "Bad Forecast Start: {} | Check Dataset Time Range".format(forecast_definition['forecast_start']))
        else:
            df = df[df[forecast_definition["time_index"]] >= forecast_definition["forecast_start"]]
            time_min = forecast_definition["forecast_start"]

    if "forecast_end" in forecast_definition:
        if pd.to_datetime(str(forecast_definition["forecast_end"])) > time_max:
            new_dates = pd.date_range(time_max, pd.to_datetime(str(forecast_definition["forecast_end"])),
                                      freq=forecast_definition["forecast_freq"])
            new_dates_df = dd.from_pandas(pd.DataFrame(new_dates, columns=[forecast_definition["time_index"]]),
                                          chunksize=10000)
            df = df.append(new_dates_df)
        else:
            df = df[df[forecast_definition["time_index"]] <= forecast_definition["forecast_end"]]
        time_max = pd.to_datetime(str(forecast_definition["forecast_end"]))

    if "joins" in forecast_definition:
        for i, join in enumerate(forecast_definition["joins"]):
            join_df = dd.read_parquet("{}/data/*".format(join["dataset_directory"]))
            df = df.merge(
                join_df,
                how="left",
                left_on=join["join_on"][0],
                right_on=join["join_on"][1],
                suffixes=("", "{}_".format(join["as"])),
            )

    if "scenarios" in forecast_definition:
        for x in forecast_definition["scenarios"]:

            scenario_ranges = [x for x in forecast_definition["scenarios"][x]["values"] if type(x) == tuple]
            if len(scenario_ranges) > 0:
                forecast_definition["scenarios"][x]["values"] = [x for x in
                                                                 forecast_definition["scenarios"][x]["values"] if
                                                                 type(x) == int]
                for y in scenario_ranges:
                    forecast_definition["scenarios"][x]["values"] = set(
                        forecast_definition["scenarios"][x]["values"] + list(range(y[0], y[1] + 1)))

            df_scenario = df[(df[forecast_definition["time_index"]] <= pd.to_datetime(
                str(forecast_definition["scenarios"][x]["end"]))) & (
                                         df[forecast_definition["time_index"]] >= pd.to_datetime(
                                     str(forecast_definition["scenarios"][x]["start"])))]
            df = df[(df[forecast_definition["time_index"]] > pd.to_datetime(
                str(forecast_definition["scenarios"][x]["end"]))) | (
                            df[forecast_definition["time_index"]] < pd.to_datetime(
                        str(forecast_definition["scenarios"][x]["start"])))]
            for v in forecast_definition["scenarios"][x]["values"]:
                df_scenario[x] = v
                df = df.append(df_scenario)

    df = cull_empty_partitions(df)
    df = df.reset_index(drop=True)
    return df


def build_dataset_dask(s3_fs, read_path, write_path, partition_dimensions=None):
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
