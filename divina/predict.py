import sys
import dask.dataframe as dd
import pandas as pd
import joblib
from .dataset import get_dataset
import backoff
from botocore.exceptions import ClientError
import os
from .utils import cull_empty_partitions


@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def dask_predict(s3_fs, forecast_definition, read_path, write_path):
    df = get_dataset(forecast_definition)

    horizon_ranges = [x for x in forecast_definition["time_horizons"] if type(x) == tuple]
    if len(horizon_ranges) > 0:
        forecast_definition["time_horizons"] = [x for x in forecast_definition["time_horizons"] if type(x) == int]
        for x in horizon_ranges:
            forecast_definition["time_horizons"] = set(forecast_definition["time_horizons"] + list(range(x[0], x[1])))

    if write_path[:5] == "s3://":
        if not s3_fs.exists(write_path):
            s3_fs.mkdir(
                write_path,
                create_parents=True,
                region_name=os.environ["AWS_DEFAULT_REGION"],
                acl="private",
            )

    for s in forecast_definition["time_validation_splits"]:

        for h in forecast_definition["time_horizons"]:
            with s3_fs.open(
                    "{}/models/s-{}_h-{}".format(
                        read_path,
                        pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
                        h,
                    ),
                    "rb",
            ) as f:
                fit_model = joblib.load(f)

            df[
                "{}_h_{}_pred".format(forecast_definition["target"], h)
            ] = fit_model.predict(
                df[
                    [
                        c
                        for c in df.columns
                        if not c
                               in [
                                   forecast_definition["time_index"],
                                   forecast_definition["target"],
                               ]
                               + [
                                   "{}_h_{}".format(forecast_definition["target"], h)
                                   for h in forecast_definition["time_horizons"]
                               ]
                    ]
                ].to_dask_array(lengths=True)
            )

            sys.stdout.write("Predictions made for horizon {}\n".format(h))

        dd.to_parquet(
            df[
                [forecast_definition["time_index"]]
                + [
                    "{}_h_{}_pred".format(forecast_definition["target"], h)
                    for h in forecast_definition["time_horizons"]
                ]
                ],
            "{}/predictions/s-{}".format(
                write_path,
                pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
            )
        )
