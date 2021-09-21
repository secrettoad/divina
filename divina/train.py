import sys
import os
import dask.dataframe as dd
import joblib
import pandas as pd
from .dataset import get_dataset
import pathlib
import backoff
from botocore.exceptions import ClientError
from dask_ml.linear_model import LinearRegression


@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def dask_train(s3_fs, forecast_definition, write_path, dask_model=LinearRegression):
    if write_path[:5] == "s3://":
        if not s3_fs.exists(write_path):
            s3_fs.mkdir(
                "{}/{}".format(write_path, "models"),
                create_parents=True,
                region_name=os.environ["AWS_DEFAULT_REGION"],
                acl="private",
            )
    else:
        pathlib.Path(os.path.join(write_path), "models").mkdir(
            parents=True, exist_ok=True
        )

    sys.stdout.write("Loading dataset\n")

    df = get_dataset(forecast_definition)

    if "signal_dimensions" in forecast_definition:
        df = df.groupby([forecast_definition["time_index"]] + forecast_definition["signal_dimensions"]).agg(
            "sum").reset_index()
    else:
        df = df.groupby(forecast_definition["time_index"]).agg("sum").reset_index()

    horizon_ranges = [x for x in forecast_definition["time_horizons"] if type(x) == tuple]
    if len(horizon_ranges) > 0:
        forecast_definition["time_horizons"] = [x for x in forecast_definition["time_horizons"] if type(x) == int]
        for x in horizon_ranges:
            forecast_definition["time_horizons"] = set(forecast_definition["time_horizons"] + list(range(x[0], x[1])))

    df[forecast_definition["time_index"]] = dd.to_datetime(
        df[forecast_definition["time_index"]], unit="s"
    )

    if "drop_features" in forecast_definition:
        df = df.drop(columns=forecast_definition["drop_features"])

    for h in forecast_definition["time_horizons"]:
        if "signal_dimension" in forecast_definition:
            df["{}_h_{}".format(forecast_definition["target"], h)] = df.groupby(
                forecast_definition["signal_dimensions"]
            )[forecast_definition["target"]].shift(-h)
        else:
            df["{}_h_{}".format(forecast_definition["target"], h)] = df[
                forecast_definition["target"]
            ].shift(-h)

    models = {}

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
        if pd.to_datetime(str(s)) <= time_min or pd.to_datetime(str(s)) >= time_max:
            raise Exception("Bad Time Split: {} | Check Dataset Time Range".format(s))
        df_train = df[df[forecast_definition["time_index"]] < s]
        for h in forecast_definition["time_horizons"]:

            model = dask_model()

            model.fit(
                df_train[
                    [
                        c
                        for c in df_train.columns
                        if not c
                        in [
                            "{}_h_{}".format(forecast_definition["target"], h)
                            for h in forecast_definition["time_horizons"]
                        ]
                        + [
                            forecast_definition["time_index"],
                            forecast_definition["target"],
                        ]
                    ]
                ].to_dask_array(lengths=True),
                df_train["{}_h_{}".format(forecast_definition["target"], h)],
            )

            sys.stdout.write("Pipeline fit for horizon {}\n".format(h))

            models[
                "s-{}_h-{}".format(pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"), h)
            ] = model

            with s3_fs.open(
                "{}/models/s-{}_h-{}".format(
                    write_path,
                    pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
                    h,
                ),
                "wb",
            ) as f:
                joblib.dump(model, f)

            sys.stdout.write("Pipeline persisted for horizon {}\n".format(h))

    return models
