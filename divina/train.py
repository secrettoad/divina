import sys
import os
import joblib
import pandas as pd
from .dataset import get_dataset
import pathlib
import backoff
from botocore.exceptions import ClientError
from dask_ml.linear_model import LinearRegression
import json
import dask.bag as db
import numpy as np


@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def dask_train(s3_fs, forecast_definition, write_path, dask_model=LinearRegression, random_seed=None):
    if write_path[:5] == "s3://":
        if not s3_fs.exists(write_path):
            s3_fs.mkdir(
                "{}/{}".format(write_path, "models/bootstrap"),
                create_parents=True,
                region_name=os.environ["AWS_DEFAULT_REGION"],
                acl="private",
            )
        write_open = s3_fs.open

    else:
        pathlib.Path(os.path.join(write_path), "models/bootstrap").mkdir(
            parents=True, exist_ok=True
        )
        write_open = open

    sys.stdout.write("Loading dataset\n")

    dataset_kwargs = {}
    for k in ['train_start', 'train_end']:
        if k in forecast_definition:
            dataset_kwargs.update({k.split('_')[1]: forecast_definition[k]})

    df = get_dataset(forecast_definition)

    for h in forecast_definition["time_horizons"]:
        if "signal_dimension" in forecast_definition:
            df["{}_h_{}".format(forecast_definition["target"], h)] = df.groupby(
                forecast_definition["signal_dimensions"]
            )[forecast_definition["target"]].shift(-h)
        else:
            df["{}_h_{}".format(forecast_definition["target"], h)] = df[
                forecast_definition["target"]
            ].shift(-h)

    time_min, time_max = (
        pd.to_datetime(str(df[forecast_definition["time_index"]].min().compute())),
        pd.to_datetime(str(df[forecast_definition["time_index"]].max().compute())),
    )

    for s in forecast_definition["time_validation_splits"]:
        if pd.to_datetime(str(s)) <= time_min or pd.to_datetime(str(s)) >= time_max:
            raise Exception("Bad Time Split: {} | Check Dataset Time Range".format(s))
        df_train = df[df[forecast_definition["time_index"]] < s]
        for h in forecast_definition["time_horizons"]:
            model = dask_model(solver_kwargs={"normalize":False})

            if "drop_features" in forecast_definition:
                features = [
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
                           + forecast_definition["drop_features"]
                ]
            else:
                features = [
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

            df_train = df_train[~df_train["{}_h_{}".format(forecast_definition["target"], h)].isnull()]

            ###TODO START HERE
            model.fit(
                df_train[features].to_dask_array(lengths=True),
                df_train["{}_h_{}".format(forecast_definition["target"], h)],
            )

            sys.stdout.write("Pipeline fit for horizon {}\n".format(h))

            with write_open(
                    "{}/models/s-{}_h-{}".format(
                        write_path,
                        pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
                        h,
                    ),
                    "wb",
            ) as f:
                joblib.dump(model, f)

            with write_open(
                    "{}/models/s-{}_h-{}_params.json".format(
                        write_path,
                        pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
                        h,
                    ),
                    "w",
            ) as f:
                json.dump({"params": {feature: coef for feature, coef in zip(features, model.coef_)}}, f)

            if 'confidence_intervals' in forecast_definition:
                if not 'bootstrap_sample' in forecast_definition:
                    bootstrap_sample = 30
                else:
                    bootstrap_sample = forecast_definition['bootstrap_sample']
                from functools import partial

                def train_persist_bootstrap_model(features, df, target, random_seed):
                    if random_seed:
                        df_train_bootstrap = df.sample(replace=False, frac=.8, random_state=random_seed)
                    else:
                        df_train_bootstrap = df.sample(replace=False, frac=.8)
                    bootstrap_model = dask_model(solver_kwargs={"normalize": False})
                    bootstrap_model.fit(
                        df_train_bootstrap[features].to_dask_array(lengths=True),
                        df_train_bootstrap[target],
                    )

                    with write_open(
                            "{}/models/bootstrap/s-{}_h-{}_r-{}".format(
                                write_path,
                                pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
                                h,
                                random_seed
                            ),
                            "wb",
                    ) as f:
                        joblib.dump(bootstrap_model, f)

                    with write_open(
                            "{}/models/bootstrap/s-{}_h-{}_r-{}_params.json".format(
                                write_path,
                                pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
                                h,
                                random_seed
                            ),
                            "w",
                    ) as f:
                        json.dump(
                            {"params": {feature: coef for feature, coef in zip(features, bootstrap_model.coef_)}}, f)

                    sys.stdout.write("Pipeline persisted for horizon {} seed {}\n".format(h, random_seed))

                    return bootstrap_model

                if random_seed:
                    sample_bag = db.from_sequence([x for x in range(random_seed, random_seed + bootstrap_sample)], npartitions=10)
                else:
                    sample_bag = db.from_sequence([x for x in np.random.randint(0, 10000, size=bootstrap_sample)], npartitions=10)
                sample_bag.map(
                    partial(train_persist_bootstrap_model, features, df_train,
                            "{}_h_{}".format(forecast_definition["target"], h))).compute()

            sys.stdout.write("Pipeline persisted for horizon {}\n".format(h))
