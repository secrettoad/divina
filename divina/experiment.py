import sys
import dask.dataframe as dd
import joblib
from .dataset import _get_dataset
import backoff
from botocore.exceptions import ClientError
import os
from functools import partial
from .utils import validate_experiment_definition, create_write_directory, cull_empty_partitions
import json
import dask.array as da
from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np
from dask_ml.linear_model import LinearRegression
import s3fs


class Experiment():
    def __init__(self, target, time_index, data_path, joins=None, time_horizons=None, train_start=None, train_end=None, forecast_start=None, forecast_end=None,
                 validate_start=None, validate_end=None, validation_splits=None, link_function=None, confidence_intervals=None, bootstrap_sample=None):
        if not time_horizons:
            self.time_horizons = [0]
        else:
            self.time_horizons = time_horizons
        if not confidence_intervals:
            self.confidence_intervals = []
        else:
            self.confidence_intervals = self.confidence_intervals
        self.train_start=train_start
        self.train_end=train_end
        self.forecast_start = forecast_start
        self.forecast_end = forecast_end
        self.validate_start = validate_start
        self.validate_end = validate_end
        self.validation_splits = validation_splits
        self.link_function = link_function
        self.target = target
        self.time_index = time_index
        self.bootstrap_sample = bootstrap_sample

    @create_write_directory
    @validate_experiment_definition
    @backoff.on_exception(backoff.expo, ClientError, max_time=30)
    def forecast(self, read_path, write_path):
        forecast_df = _get_dataset(pad=True, start=self.forecast_start, end=self.forecast_end)

        horizon_ranges = [x for x in self.time_horizons if type(x) == tuple]
        if len(horizon_ranges) > 0:
            time_horizons = [x for x in self.time_horizons if type(x) == int]
            for x in horizon_ranges:
                time_horizons = set(time_horizons + list(range(x[0], x[1])))

        if read_path[:5] == "s3://":
            s3_fs = s3fs.S3FileSystem()
            read_open = s3_fs.open
            bootstrap_prefix = None
            read_ls = s3_fs.ls
        else:
            read_open = open
            read_ls = os.listdir
            bootstrap_prefix = os.path.join(read_path, 'models', 'bootstrap')

        for h in self.time_horizons:
            with read_open(
                    "{}/models/h-{}".format(
                        read_path,
                        h,
                    ),
                    "rb",
            ) as f:
                fit_model = joblib.load(f)
            with read_open(
                    "{}/models/h-{}_params.json".format(
                        read_path,
                        h,
                    ),
                    "r",
            ) as f:
                fit_model_params = json.load(f)
            features = fit_model_params["features"]

            for f in features:
                if not is_numeric_dtype(forecast_df[f].dtype):
                    try:
                        forecast_df[f] = forecast_df[f].astype(float)
                    except ValueError:
                        raise ValueError(
                            '{} could not be converted to float. '
                            'Please convert to numeric or encode with '
                            '"encode_features: {}"'.format(
                                f, f))

            if self.link_function:
                forecast_df[
                    "{}_h_{}_pred".format(self.target, h)
                ] = da.expm1(fit_model.predict(forecast_df[features].to_dask_array(lengths=True)))
                sys.stdout.write("Forecasts made for horizon {}\n".format(h))
            else:
                forecast_df[
                    "{}_h_{}_pred".format(self.target, h)
                ] = fit_model.predict(forecast_df[features].to_dask_array(lengths=True))
                sys.stdout.write("Forecasts made for horizon {}\n".format(h))

            factor_df = dd.from_array(
                forecast_df[features].to_dask_array(lengths=True) * da.from_array(fit_model.coef_))
            factor_df.columns = ["factor_{}".format(c) for c in features]
            for c in factor_df:
                forecast_df[c] = factor_df[c]

            if len(self.confidence_intervals) > 0:
                bootstrap_model_paths = [p for p in read_ls("{}/models/bootstrap".format(
                    read_path
                )) if '.' not in p]
                bootstrap_model_paths.sort()

                def load_and_predict_bootstrap_model(paths, df):
                    for path in paths:
                        if bootstrap_prefix:
                            model_path = os.path.join(bootstrap_prefix, path)
                        else:
                            model_path = path
                        with read_open(
                                model_path,
                                "rb",
                        ) as f:
                            bootstrap_model = joblib.load(f)
                        with read_open(
                                "{}_params.json".format(
                                    model_path),
                                "r",
                        ) as f:
                            bootstrap_params = json.load(f)
                            bootstrap_features = bootstrap_params['features']
                        if self.link_function == 'log':
                            df['{}_h_{}_pred_b_{}'.format(self.target, h,
                                                          path.split("-")[-1])] = da.expm1(
                                bootstrap_model.predict(
                                    dd.from_pandas(df[bootstrap_features], chunksize=10000).to_dask_array(
                                        lengths=True)))
                        else:
                            df['{}_h_{}_pred_b_{}'.format(self.target, h,
                                                          path.split("-")[-1])] = bootstrap_model.predict(
                                dd.from_pandas(df[bootstrap_features], chunksize=10000).to_dask_array(lengths=True))

                    return df

                forecast_df = forecast_df.map_partitions(partial(load_and_predict_bootstrap_model,
                                                                 bootstrap_model_paths
                                                                 ))

                df_interval = dd.from_array(dd.from_array(
                    forecast_df[
                        ['{}_h_{}_pred_b_{}'.format(self.target, h, i.split("-")[-1]) for i in
                         bootstrap_model_paths] + [
                            '{}_h_{}_pred'.format(self.target,
                                                  h)]].to_dask_array(lengths=True).T).repartition(
                    npartitions=forecast_df.npartitions).quantile(
                    [i * .01 for i in self.confidence_intervals]).to_dask_array(
                    lengths=True).T).persist()
                df_interval.columns = ['{}_h_{}_pred_c_{}'.format(self.target, h, c) for c in
                                       self.confidence_intervals]

                df_interval = df_interval.repartition(divisions=forecast_df.divisions).reset_index(drop=True)
                forecast_df = forecast_df.reset_index(drop=True).join(df_interval)

            forecast_df[self.time_index] = dd.to_datetime(
                forecast_df[self.time_index])

            forecast_df = forecast_df.sort_values(self.time_index)

            dd.to_parquet(
                forecast_df,
                "{}/forecast".format(
                    write_path,
                )
            )

    def _train_model(self, df, model_name, random_state, features, target,
                     link_function, write_open, write_path, bootstrap_sample=None, confidence_intervals=None):
        if random_state:
            model = LinearRegression(random_state=random_state)
        else:
            model = LinearRegression()

        df_train = df[~df[target].isnull()]
        df_std = df_train[features].std().compute()
        constant_columns = [c for c in df_std.index if df_std[c] == 0]
        features = [
            c
            for c in features
            if not c
                   in constant_columns and is_numeric_dtype(df_train.dtypes[c])
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
            json.dump({"features": features}, f)

        sys.stdout.write("Model persisted: {}\n".format(model_name))

        if confidence_intervals:
            if not bootstrap_sample:
                bootstrap_sample = 30

            def train_persist_bootstrap_model(features, df, target, link_function, model_name, random_state):

                if random_state:
                    df_train_bootstrap = df.sample(replace=False, frac=.8, random_state=random_state)
                else:
                    df_train_bootstrap = df.sample(replace=False, frac=.8)
                if random_state:
                    bootstrap_model = LinearRegression(random_state=random_state)
                else:
                    bootstrap_model = LinearRegression()
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
                            random_state
                        ),
                        "wb",
                ) as f:
                    joblib.dump(bootstrap_model, f)

                with write_open(
                        "{}/models/bootstrap/{}_r-{}_params.json".format(
                            write_path,
                            model_name,
                            random_state
                        ),
                        "w",
                ) as f:
                    json.dump(
                        {"features": bootstrap_features}, f)

                sys.stdout.write("Model persisted: {}_r-{}\n".format(model_name, random_state))

                return (bootstrap_model, bootstrap_features)

            if random_state:
                states = [x for x in range(random_state, random_state + bootstrap_sample)]
            else:
                states = [x for x in np.random.randint(0, 10000, size=bootstrap_sample)]

            bootstrap_models = []
            for state in states:
                bootstrap_models.append(
                    train_persist_bootstrap_model(features, df_train,
                                                  target,
                                                  link_function,
                                                  model_name, state))

            return (model, features), bootstrap_models

        else:

            return (model, features)

    @create_write_directory
    @validate_experiment_definition
    @backoff.on_exception(backoff.expo, ClientError, max_time=30)
    def train(self, write_path, random_state=None):

        if write_path[:5] == "s3://":
            s3_fs = s3fs.S3FileSystem()
            write_open = s3_fs.open
        else:
            write_open = open

        sys.stdout.write("Loading dataset\n")

        dataset_kwargs = {}
        for k in [self.train_start, self.train_end]:
            dataset_kwargs.update({k.split('_')[1]: k})

        df = _get_dataset()

        time_min, time_max = (
            pd.to_datetime(str(df[self.time_index].min().compute())),
            pd.to_datetime(str(df[self.time_index].max().compute())),
        )

        features = [
            c
            for c in df.columns
            if not c
                   in [
                       "{}_h_{}".format(self.target, h)
                       for h in self.time_horizons
                   ]
                   + [
                       self.time_index,
                       self.target,
                   ]
        ]

        for h in self.time_horizons:

            model, bootstrap_models = self._train_model(df=df, model_name="h-{}".format(h), random_state=random_state,
                                                        features=features, target=self.target,
                                                        bootstrap_sample=self.bootstrap_sample,
                                                        confidence_intervals=self.confidence_intervals,
                                                        link_function=self.link_function, write_open=write_open,
                                                        write_path=write_path)

            for s in self.validation_splits:
                if pd.to_datetime(str(s)) <= time_min or pd.to_datetime(str(s)) >= time_max:
                    raise Exception("Bad Time Split: {} | Check Dataset Time Range".format(s))
                df_train = df[df[self.time_index] < s]

                split_model = self._train_model(df=df_train, model_name="s-{}_h-{}".format(
                    pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
                    h
                ), random_state=random_state,
                                                features=features, target=self.target,
                                                link_function=self.link_function, write_open=write_open,
                                                write_path=write_path)

    @create_write_directory
    @validate_experiment_definition
    @backoff.on_exception(backoff.expo, ClientError, max_time=30)
    def validate(self, experiment_definition, write_path, read_path):
        def get_metrics(df):
            metrics = {"time_horizons": {}}
            for h in time_horizons:
                metrics["time_horizons"][h] = {}
                df["resid_h_{}".format(h)] = (
                        df[experiment_definition["target"]].shift(-h)
                        - df["{}_h_{}_pred".format(experiment_definition["target"], h)]
                )
                metrics["time_horizons"][h]["mae"] = (
                    df[
                        "resid_h_{}".format(h)
                    ]
                        .abs()
                        .mean()
                        .compute()
                )
            return metrics

        if read_path[:5] == "s3://":
            s3_fs = s3fs.S3FileSystem()
            read_open = s3_fs.open
            read_ls = s3_fs.ls
            write_open = s3_fs.open
            bootstrap_prefix = None
        else:
            read_open = open
            read_ls = os.listdir
            write_open = open
            bootstrap_prefix = os.path.join(read_path, 'models', 'bootstrap')

        dataset_kwargs = {}
        for k in [self.validate_start, self.validate_end]:
            if k in experiment_definition:
                dataset_kwargs.update({k.split('_')[1]: experiment_definition[k]})

        df = _get_dataset(experiment_definition)

        metrics = {"splits": {}}

        horizon_ranges = [x for x in self.time_horizons if type(x) == tuple]
        if len(horizon_ranges) > 0:
            time_horizons = [x for x in self.time_horizons if type(x) == int]
            for x in horizon_ranges:
                time_horizons = set(time_horizons + list(range(x[0], x[1])))

        df = df[
            [experiment_definition["target"], experiment_definition["time_index"]]
        ]

        time_min, time_max = (
            df[experiment_definition["time_index"]].min().compute(),
            df[experiment_definition["time_index"]].max().compute(),
        )

        for s in self.validation_splits:

            validate_kwargs = {}
            if self.validate_start:
                if pd.to_datetime(str(s)) < pd.to_datetime(str(self.validate_start)):
                    start = pd.to_datetime(str(s))
                else:
                    start = pd.to_datetime(str(self.validate_start))
            else:
                start = pd.to_datetime(str(s))
            if self.validate_end:
                if pd.to_datetime(str(s)) > pd.to_datetime(str(self.validate_end)):
                    raise Exception(
                        "Bad End: {} | Check Dataset Time Range".format(
                            pd.to_datetime(str(self.forecast_start))))
                else:
                    end = pd.to_datetime(str(s))
            else:
                end = None
            validate_df = _get_dataset(experiment_definition, pad=False, start=start, end=end)

            for h in self.time_horizons:
                with read_open(
                        "{}/models/s-{}_h-{}".format(
                            read_path,
                            pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
                            h,
                        ),
                        "rb",
                ) as f:
                    fit_model = joblib.load(f)
                with read_open(
                        "{}/models/s-{}_h-{}_params.json".format(
                            read_path,
                            pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
                            h,
                        ),
                        "r",
                ) as f:
                    fit_model_params = json.load(f)
                features = fit_model_params["features"]
                for f in features:
                    if not f in validate_df.columns:
                        validate_df[f] = 0

                if self.link_function == 'log':
                    validate_df[
                            "{}_h_{}_pred".format(self.target, h)
                        ] = da.expm1(fit_model.predict(validate_df[features].to_dask_array(lengths=True)))
                    sys.stdout.write("Validation predictions made for split {}\n".format(s))

                else:
                    validate_df[
                        "{}_h_{}_pred".format(self.target, h)
                    ] = fit_model.predict(validate_df[features].to_dask_array(lengths=True))
                    sys.stdout.write("Validation predictions made for split {}\n".format(s))

                if not pd.to_datetime(str(time_min)) < pd.to_datetime(str(s)) < pd.to_datetime(str(time_max)):
                    raise Exception("Bad Validation Split: {} | Check Dataset Time Range".format(s))

                validate_df = cull_empty_partitions(validate_df)
                metrics["splits"][s] = get_metrics(validate_df)

                with write_open("{}/metrics.json".format(write_path), "w") as f:
                    json.dump(metrics, f)

            validate_df[self.time_index] = dd.to_datetime(
                validate_df[self.time_index])

            dd.to_parquet(
                validate_df[
                    [self.time_index]
                    + [
                        "{}_h_{}_pred".format(self.target, h)
                        for h in self.time_horizons
                    ]
                    ],
                "{}/validation/s-{}".format(
                    write_path,
                    pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
                )
            )

    @validate_experiment_definition
    def _experiment(self, experiment_definition, read_path, write_path, random_state=None, s3_fs=None):
        self.train(
            s3_fs=s3_fs,
            experiment_definition=experiment_definition,
            write_path=write_path,
            random_state=random_state
        )
        self.forecast(
            s3_fs=s3_fs,
            experiment_definition=experiment_definition,
            read_path=read_path,
            write_path=write_path
        )
        self.validate(
            s3_fs=s3_fs,
            experiment_definition=experiment_definition,
            read_path=read_path,
            write_path=write_path
        )
