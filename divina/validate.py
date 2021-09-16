import json
import dask.dataframe as dd
import pandas as pd
from .dataset import get_dataset
import os
import backoff
from botocore.exceptions import ClientError


@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def dask_validate(s3_fs, forecast_definition, write_path, read_path):
    def get_metrics(forecast_definition, df, s):
        metrics = {"time_horizons": {}}
        for h in forecast_definition["time_horizons"]:
            metrics["time_horizons"][h] = {}
            df["resid_h_{}".format(h)] = (
                df[forecast_definition["target"]].shift(-h)
                - df["{}_h_{}_pred".format(forecast_definition["target"], h)]
            )
            metrics["time_horizons"][h]["mae"] = (
                df[dd.to_datetime(df[forecast_definition["time_index"]], unit="s") > s][
                    "resid_h_{}".format(h)
                ]
                .abs()
                .mean()
                .compute()
            )
        return metrics

    if write_path[:5] == "s3://":
        if not s3_fs.exists(write_path):
            s3_fs.mkdir(
                write_path,
                create_parents=True,
                region_name=os.environ["AWS_DEFAULT_REGION"],
                acl="private",
            )

    metrics = {"splits": {}}

    horizon_ranges = [x for x in forecast_definition["time_horizons"] if type(x) == tuple]
    if len(horizon_ranges) > 0:
        forecast_definition["time_horizons"] = [x for x in forecast_definition["time_horizons"] if type(x) == int]
        for x in horizon_ranges:
            forecast_definition["time_horizons"] = set(forecast_definition["time_horizons"] + list(range(x[0], x[1])))

    df = get_dataset(forecast_definition)

    if "signal_dimensions" in forecast_definition:
        df = df.groupby([forecast_definition["time_index"]] + forecast_definition["signal_dimensions"]).agg(
            "sum").reset_index()
    else:
        df = df.groupby(forecast_definition["time_index"]).agg("sum").reset_index()

    df = df[
        [forecast_definition["target"], forecast_definition["time_index"]]
    ]

    df[forecast_definition["time_index"]] = dd.to_datetime(
        df[forecast_definition["time_index"]], unit="s"
    )

    time_min, time_max = (
        df[forecast_definition["time_index"]].min().compute(),
        df[forecast_definition["time_index"]].max().compute(),
    )

    if "train_validation_cutoff" in forecast_definition:
        if pd.to_datetime(str(forecast_definition["train_end"])) > time_max:
            raise Exception("Bad Train End: {} | Check Dataset Time Range".format(forecast_definition['train_end']))
        else:
            df = df[df[forecast_definition["time_index"]] <= forecast_definition["train_end"]]
            time_max = pd.to_datetime(str(forecast_definition["train_end"]))

    for s in forecast_definition["time_validation_splits"]:

        if not time_min < pd.to_datetime(str(s)) < time_max:
            raise Exception("Bad Validation Split: {} | Check Dataset Time Range".format(s))

        df_pred = dd.read_parquet(
            "{}/predictions/s-{}/*".format(
                read_path, pd.to_datetime(s).strftime("%Y%m%d-%H%M%S")
            )
        )

        if "signal_dimensions" in forecast_definition:
            df = df_pred.merge(
                df,
                on=[forecast_definition["time_index"]]
                + forecast_definition["signal_dimensions"],
            )
        else:
            df = df_pred.merge(df, on=[forecast_definition["time_index"]])
        metrics["splits"][s] = get_metrics(forecast_definition, df, s)

        with s3_fs.open("{}/metrics.json".format(write_path), "w") as f:
            json.dump(metrics, f)
