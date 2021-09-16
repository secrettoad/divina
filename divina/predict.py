import sys
import dask.dataframe as dd
import pandas as pd
import joblib
from .dataset import get_dataset
import backoff
from botocore.exceptions import ClientError
import os


@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def dask_predict(s3_fs, forecast_definition, read_path, write_path):
    df = get_dataset(forecast_definition)

    df[forecast_definition["time_index"]] = dd.to_datetime(
        df[forecast_definition["time_index"]], unit="s"
    )
    if "drop_features" in forecast_definition:
        df = df.drop(columns=forecast_definition["drop_features"])

    if "signal_dimensions" in forecast_definition:
        df = df.groupby([forecast_definition["time_index"]] + forecast_definition["signal_dimensions"]).agg(
            "sum").reset_index()
    else:
        df = df.groupby(forecast_definition["time_index"]).agg("sum").reset_index()

    if "scenarios" in forecast_definition:
        for x in forecast_definition["scenarios"]:

            scenario_ranges = [x for x in forecast_definition["scenarios"][x]["values"] if type(x) == tuple]
            if len(scenario_ranges) > 0:
                forecast_definition["scenarios"][x]["values"] = [x for x in forecast_definition["scenarios"][x]["values"] if type(x) == int]
                for y in scenario_ranges:
                    forecast_definition["scenarios"][x]["values"] = set(
                        forecast_definition["scenarios"][x]["values"] + list(range(y[0], y[1]+1)))

            df_scenario = df[(df[forecast_definition["time_index"]] <= pd.to_datetime(
                                          str(forecast_definition["scenarios"][x]["end"]))) & (df[forecast_definition["time_index"]] >= pd.to_datetime(
                                          str(forecast_definition["scenarios"][x]["start"])))]
            df = df[(df[forecast_definition["time_index"]] > pd.to_datetime(
                str(forecast_definition["scenarios"][x]["end"]))) | (
                                         df[forecast_definition["time_index"]] < pd.to_datetime(
                                     str(forecast_definition["scenarios"][x]["start"])))]
            for v in forecast_definition["scenarios"][x]["values"]:
                df_scenario[x] = v
                df = df.append(df_scenario)

    horizon_ranges = [x for x in forecast_definition["time_horizons"] if type(x) == tuple]
    if len(horizon_ranges) > 0:
        forecast_definition["time_horizons"] = [x for x in forecast_definition["time_horizons"] if type(x) == int]
        for x in horizon_ranges:
            forecast_definition["time_horizons"] = set(forecast_definition["time_horizons"] + list(range(x[0], x[1])))

    time_min, time_max = (
        df[forecast_definition["time_index"]].min().compute(),
        df[forecast_definition["time_index"]].max().compute(),
    )

    if "forecast_start" in forecast_definition:
        if pd.to_datetime(str(forecast_definition["forecast_start"])) < time_min:
            raise Exception(
                "Bad Forecast Start: {} | Check Dataset Time Range".format(forecast_definition['forecast_start']))
        else:
            df = df[df[forecast_definition["time_index"]] >= forecast_definition["forecast_start"]]
            time_min = pd.to_datetime(str(forecast_definition["forecast_start"]))

    if "forecast_end" in forecast_definition:
        if pd.to_datetime(str(forecast_definition["forecast_end"])) > time_max:
            raise Exception(
                "Bad Forecast End: {} | Check Dataset Time Range".format(forecast_definition['forecast_end']))
        else:
            df = df[df[forecast_definition["time_index"]] <= forecast_definition["forecast_end"]]
            time_max = pd.to_datetime(str(forecast_definition["forecast_end"]))

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
