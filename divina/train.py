import sys
import os
import joblib
import pandas as pd
from .dataset import _get_dataset
import pathlib
import backoff
from botocore.exceptions import ClientError
import json
import dask.bag as db
import numpy as np
from .utils import validate_forecast_definition
from dask_ml.linear_model import LinearRegression
import dask.array as da
from functools import partial


def _train_model(df, dask_model, model_name, random_seed, features, target, bootstrap_sample, confidence_intervals,
                 link_function, write_open, write_path):
    if random_seed:
        model = dask_model(random_state=random_seed)
    else:
        model = dask_model()

    df_train = df[~df[target].isnull()]
    df_std = df_train[features].std().compute()
    constant_columns = [c for c in df_std.index if df_std[c] == 0]
    features = [
        c
        for c in features
        if not c
               in constant_columns
    ]

    if link_function == "log":
        model.fit(
            df_train[features].to_dask_array(lengths=True),
            da.log1p(
                df_train[target].to_dask_array(lengths=True)),
        )
    else:
        model.fit(
            df_train[features].to_dask_array(lengths=True),
            df_train[target].to_dask_array(lengths=True))
    with write_open(
            "{}/models/{}".format(
                write_path,
                model_name
            ),
            "wb",
    ) as f:
        joblib.dump(model, f)

    with write_open(
            "{}/models/{}_params.json".format(
                write_path,
                model_name
            ),
            "w",
    ) as f:
        json.dump({"params": {feature: coef for feature, coef in zip(features, model.coef_)},
                   "intercept": model.intercept_}, f)

    sys.stdout.write("Model persisted: {}\n".format(model_name))

    if confidence_intervals:
        if not bootstrap_sample:
            bootstrap_sample = 30

        def train_persist_bootstrap_model(features, df, target, link_function, model_name, random_seed):

            if random_seed:
                df_train_bootstrap = df.sample(replace=False, frac=.8, random_state=random_seed)
            else:
                df_train_bootstrap = df.sample(replace=False, frac=.8)
            if random_seed:
                bootstrap_model = dask_model(random_state=random_seed)
            else:
                bootstrap_model = dask_model()
            df_std_bootstrap = df_train_bootstrap[features].std().compute()
            bootstrap_features = [c for c in features if not df_std_bootstrap.loc[c] == 0]
            if link_function == 'log':
                bootstrap_model.fit(
                    df_train_bootstrap[bootstrap_features].to_dask_array(lengths=True),
                    da.log1p(df_train_bootstrap[target].to_dask_array(lengths=True)),
                )
            else:
                bootstrap_model.fit(
                    df_train_bootstrap[bootstrap_features].to_dask_array(lengths=True),
                    df_train_bootstrap[target].to_dask_array(lengths=True),
                )
            with write_open(
                    "{}/models/bootstrap/{}_r-{}".format(
                        write_path,
                        model_name,
                        random_seed
                    ),
                    "wb",
            ) as f:
                joblib.dump(bootstrap_model, f)

            with write_open(
                    "{}/models/bootstrap/{}_r-{}_params.json".format(
                        write_path,
                        model_name,
                        random_seed
                    ),
                    "w",
            ) as f:
                json.dump(
                    {"params": {feature: coef for feature, coef in zip(features, bootstrap_model.coef_)},
                     'intercept': model.intercept_}, f)

            sys.stdout.write("Model persisted: {}_r-{}\n".format(model_name, random_seed))

            return (bootstrap_model, bootstrap_features)

        if random_seed:
            sample_bag = db.from_sequence([x for x in range(random_seed, random_seed + bootstrap_sample)],
                                          npartitions=10)
        else:
            sample_bag = db.from_sequence([x for x in np.random.randint(0, 10000, size=bootstrap_sample)],
                                          npartitions=10)
        bootstrap_models = sample_bag.map(
            partial(train_persist_bootstrap_model, features, df_train,
                    target,
                    link_function,
                    model_name)).compute()

        return model, bootstrap_models


@validate_forecast_definition
@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def _train(s3_fs, forecast_definition, write_path, dask_model=LinearRegression, random_seed=None):
    if "model" in forecast_definition:
        dask_model = globals()[forecast_definition["model"]]
    if not "confidence_intervals" in forecast_definition:
        confidence_intervals = None
    else:
        confidence_intervals = forecast_definition["confidence_intervals"]
    if not "bootstrap_sample" in forecast_definition:
        bootstrap_sample = None
    else:
        bootstrap_sample = forecast_definition["bootstrap_sample"]
    if not "link_function" in forecast_definition:
        link_function = None
    else:
        link_function = forecast_definition["link_function"]
    if write_path[:5] == "s3://":
        if not s3_fs.exists(write_path):
            s3_fs.mkdir(
                write_path,
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

    df = _get_dataset(forecast_definition)

    for h in forecast_definition["time_horizons"]:
        if "signal_dimension" in forecast_definition:
            df["{}_h_{}".format(forecast_definition["target"], h)] = df.groupby(
                forecast_definition["signal_dimensions"]
            )[forecast_definition["target"]].apply(lambda x: x.shift(-h))
        else:
            df["{}_h_{}".format(forecast_definition["target"], h)] = df[
                forecast_definition["target"]
            ].shift(-h)

    time_min, time_max = (
        pd.to_datetime(str(df[forecast_definition["time_index"]].min().compute())),
        pd.to_datetime(str(df[forecast_definition["time_index"]].max().compute())),
    )

    features = [
        c
        for c in df.columns
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

    for h in forecast_definition["time_horizons"]:

        model, bootstrap_models = _train_model(df=df, dask_model=dask_model, model_name="h-{}".format(h), random_seed=random_seed,
                     features=features, target=forecast_definition["target"],
                     bootstrap_sample=bootstrap_sample, confidence_intervals=confidence_intervals,
                     link_function=link_function, write_open=write_open, write_path=write_path)

        for s in forecast_definition["time_validation_splits"]:
            if pd.to_datetime(str(s)) <= time_min or pd.to_datetime(str(s)) >= time_max:
                raise Exception("Bad Time Split: {} | Check Dataset Time Range".format(s))
            df_train = df[df[forecast_definition["time_index"]] < s]

            model, bootstrap_models = _train_model(df=df_train, dask_model=dask_model, model_name="s-{}_h-{}".format(
                pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
                h
            ), random_seed=random_seed,
                         features=features, target="{}_h_{}".format(forecast_definition["target"], h),
                         bootstrap_sample=bootstrap_sample, confidence_intervals=confidence_intervals,
                         link_function=link_function, write_open=write_open, write_path=write_path)
