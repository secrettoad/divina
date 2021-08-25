import json
import dask.dataframe as dd
import pandas as pd
from .dataset import get_dataset
import os
import backoff
from botocore.exceptions import ClientError

@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def dask_validate(s3_fs, vision_definition, write_path, read_path, vision_id):
    def get_metrics(vision_definition, df, s):
        metrics = {"time_horizons": {}}
        for h in vision_definition["time_horizons"]:
            metrics["time_horizons"][h] = {}
            df["resid_h_{}".format(h)] = (
                df[vision_definition["target"]].shift(-h)
                - df["{}_h_{}_pred".format(vision_definition["target"], h)]
            )
            metrics["time_horizons"][h]["mae"] = (
                df[dd.to_datetime(df[vision_definition["time_index"]], unit="s") > s][
                    "resid_h_{}".format(h)
                ]
                .abs()
                .mean()
                .compute()
            )
        return metrics

    if write_path[:5] == "s3://":
        if not s3_fs.exists(write_path):
            s3_fs.mkdir(write_path, create_parents=True, region_name=os.environ["AWS_DEFAULT_REGION"], acl='private')

    metrics = {"splits": {}}
    for s in vision_definition["time_validation_splits"]:

        df_pred = dd.read_parquet(
            "{}/{}/predictions/s-{}/*".format(
                read_path, vision_id, pd.to_datetime(s).strftime("%Y%m%d-%H%M%S")
            )
        )

        df_base, profile_base = get_dataset(vision_definition)
        df_base = df_base[
            [vision_definition["target"], vision_definition["time_index"]]
        ]

        if "signal_dimensions" in vision_definition:
            df = df_pred.merge(
                df_base,
                on=[vision_definition["time_index"]]
                + vision_definition["signal_dimensions"],
            )
        else:
            df = df_pred.merge(df_base, on=[vision_definition["time_index"]])
        metrics["splits"][s] = get_metrics(vision_definition, df, s)

        with s3_fs.open(
            "{}/{}/metrics.json".format(write_path, vision_id), "w"
        ) as f:
            json.dump(metrics, f)
