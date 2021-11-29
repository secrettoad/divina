import sys
import dask.dataframe as dd
import joblib
import backoff
from botocore.exceptions import ClientError
from .datasets.load import _load
import os
from functools import partial
from .utils import create_write_directory, cull_empty_partitions
import json
import dask.array as da
from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np
from dask_ml.linear_model import LinearRegression
import s3fs
from dask_ml.preprocessing import Categorizer, DummyEncoder
from sklearn.pipeline import make_pipeline
from itertools import product


class Experiment():
    def __init__(self, target, time_index, data_path, target_dimensions=None, include_features=None, drop_features=None, joins=None, encode_features=None, bin_features=None, interaction_features=None, time_horizons=None, train_start=None, train_end=None, forecast_start=None, forecast_end=None,
                 validate_start=None, validate_end=None, validation_splits=None, link_function=None, confidence_intervals=None, bootstrap_sample=None, scenarios=None, frequency=None):
        if not time_horizons:
            self.time_horizons = [0]
        else:
            horizon_ranges = [x for x in time_horizons if type(x) == tuple]
            if len(horizon_ranges) > 0:
                self.time_horizons = [x for x in time_horizons if type(x) == int]
                for x in horizon_ranges:
                    self.time_horizons = set(self.time_horizons + list(range(x[0], x[1])))
            self.time_horizons = time_horizons
        if not confidence_intervals:
            self.confidence_intervals = []
        else:
            self.confidence_intervals = confidence_intervals
        self.data_path = data_path
        self.target_dimensions = target_dimensions
        self.joins = joins
        self.include_features = include_features
        self.drop_features = drop_features
        self.encode_features = encode_features
        self.bin_features = bin_features
        self.interaction_features = interaction_features
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
        self.scenarios = scenarios
        self.frequency = frequency
        self.models = {}
        self.metrics = {}

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

            bootstrap_models = {}
            for state in states:
                bootstrap_models[state] = train_persist_bootstrap_model(features, df_train,
                                                  target,
                                                  link_function,
                                                  model_name, state)

            return (model, features), bootstrap_models

        else:

            return (model, features)

    @create_write_directory
    @backoff.on_exception(backoff.expo, ClientError, max_time=30)
    def train(self, write_path, random_state=None):

        if write_path[:5] == "s3://":
            s3_fs = s3fs.S3FileSystem()
            write_open = s3_fs.open
        else:
            write_open = open

        sys.stdout.write("Loading dataset\n")

        df = self.get_dataset(start=self.train_start, end=self.train_end)

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

        self.models = {'horizons': {h: {} for h in self.time_horizons}}
        self.validation_models = {'horizons': {h: {} for h in self.time_horizons}}

        for h in self.time_horizons:

            model, bootstrap_models = self._train_model(df=df, model_name="h-{}".format(h), random_state=random_state,
                                                        features=features, target=self.target,
                                                        bootstrap_sample=self.bootstrap_sample,
                                                        confidence_intervals=self.confidence_intervals,
                                                        link_function=self.link_function, write_open=write_open,
                                                        write_path=write_path)

            self.models['horizons'][h]['base'] = model
            self.models['horizons'][h]['bootstrap'] = bootstrap_models

            if self.validation_splits:
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

                    self.models['horizons'][h]['splits'] = {}
                    self.models['horizons'][h]['splits'][s] = split_model

    @create_write_directory
    @backoff.on_exception(backoff.expo, ClientError, max_time=30)
    def forecast(self, read_path, write_path):
        forecast_df = self.get_dataset(start=self.forecast_start, end=self.forecast_end)

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

                def load_and_predict_bootstrap_model(paths, target, link_function, df):
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
                        if link_function == 'log':
                            df['{}_h_{}_pred_b_{}'.format(target, h,
                                                          path.split("-")[-1])] = da.expm1(
                                bootstrap_model.predict(
                                    dd.from_pandas(df[bootstrap_features], chunksize=10000).to_dask_array(
                                        lengths=True)))
                        else:
                            df['{}_h_{}_pred_b_{}'.format(target, h,
                                                          path.split("-")[-1])] = bootstrap_model.predict(
                                dd.from_pandas(df[bootstrap_features], chunksize=10000).to_dask_array(lengths=True))

                    return df

                forecast_df = forecast_df.map_partitions(partial(load_and_predict_bootstrap_model,
                                                                 bootstrap_model_paths, self.target, self.link_function
                                                                 ))

                ###TODO rewrite this....horrendously slow and not distributing to more than one worker

                df_interval = dd.from_array(dd.from_array(
                    forecast_df[
                        ['{}_h_{}_pred_b_{}'.format(self.target, h, i.split("-")[-1]) for i in
                         bootstrap_model_paths] + [
                            '{}_h_{}_pred'.format(self.target,
                                                  h)]].to_dask_array(lengths=True).T).repartition(
                    npartitions=forecast_df.npartitions).quantile(
                    [i * .01 for i in self.confidence_intervals]).to_dask_array(
                    lengths=True).T)
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

            return forecast_df

    @create_write_directory
    @backoff.on_exception(backoff.expo, ClientError, max_time=30)
    def validate(self, write_path, read_path):
        def get_metrics(df):
            metrics = {"time_horizons": {}}
            for h in self.time_horizons:
                metrics["time_horizons"][h] = {}
                df["resid_h_{}".format(h)] = (
                        df[self.target].shift(-h)
                        - df["{}_h_{}_pred".format(self.target, h)]
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
            write_open = s3_fs.open
        else:
            read_open = open
            write_open = open

        df = self.get_dataset()

        metrics = {"splits": {}}

        df = df[
            [self.target, self.time_index]
        ]

        time_min, time_max = (
            df[self.time_index].min().compute(),
            df[self.time_index].max().compute(),
        )

        del(df)

        for s in self.validation_splits:

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
            validate_df = self.get_dataset(start=start, end=end)

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

                self.metrics = metrics

                with write_open("{}/metrics.json".format(write_path), "w") as f:
                    json.dump(metrics, f)

            validate_df[self.time_index] = dd.to_datetime(
                validate_df[self.time_index])

            self.validation = validate_df.compute()

    @backoff.on_exception(backoff.expo, ClientError, max_time=30)
    def get_dataset(self, start=None, end=None):
        if self.data_path.startswith("divina://"):
            df = _load(self.data_path)
        else:
            df = dd.read_parquet("{}/*".format(self.data_path))
        npartitions = (df.memory_usage(deep=True).sum().compute() // 104857600) + 1

        df[self.time_index] = dd.to_datetime(df[self.time_index])

        time_min, time_max = (
            df[self.time_index].min().compute(),
            df[self.time_index].max().compute(),
        )

        if self.target_dimensions:
            df = df.groupby([self.time_index] + self.target_dimensions).agg(
                {**{c: "sum" for c in df.columns if
                    df[c].dtype in [int, float] and c != self.time_index},
                 **{c: "first" for c in df.columns if
                    df[c].dtype not in [int, float] and c != self.time_index}}).drop(
                columns=self.target_dimensions).reset_index()
        else:
            df = df.groupby(self.time_index).agg(
                {**{c: "sum" for c in df.columns if
                    df[c].dtype in [int, float] and c != self.time_index},
                 **{c: "first" for c in df.columns if
                    df[c].dtype not in [int, float] and c != self.time_index}}).reset_index()

        if start:
            if pd.to_datetime(start) < time_min:
                raise Exception(
                    "Bad Start: {} < {} Check Dataset Time Range".format(start, time_min))
            else:
                df = df[dd.to_datetime(df[self.time_index]) >= start]
                time_min = pd.to_datetime(str(start))

        if end:
            if pd.to_datetime(end) > time_max:
                if not self.scenarios:
                    raise Exception(
                        "Bad End: {} | {} Check Dataset Time Range".format(end, time_max))
            else:
                df = df[dd.to_datetime(df[self.time_index]) <= end]
                time_max = pd.to_datetime(str(end))

        if self.scenarios:
            if not self.frequency:
                raise Exception(
                    'Frequency of time series must be supplied. Please supply with "frequency: "D", "M", "s", etc."')
            if end:
                if start and pd.to_datetime(start) > time_max:
                    new_dates = pd.date_range(pd.to_datetime(str(start)), pd.to_datetime(str(end)),
                                              freq=self.frequency)
                else:
                    new_dates = pd.date_range(
                        time_max + pd.tseries.frequencies.to_offset(self.frequency),
                        pd.to_datetime(str(end)),
                        freq=self.frequency)
                if len(new_dates) > 0:

                    combinations = list(new_dates)
                    if self.target_dimensions:
                        combinations = [list(x) for x in product(combinations,
                                                                 *[df[s].unique().compute().values for s in
                                                                   self.target_dimensions])]
                        scenario_columns = [self.time_index] + self.target_dimensions
                    else:
                        combinations = [[x] for x in combinations]
                        scenario_columns = [self.time_index]
                    constant_columns = [c for c in self.scenarios if
                                        self.scenarios[c]["mode"] == "constant"]
                    for c in constant_columns:
                        combinations = [x[0] + [x[1]] for x in
                                        product(combinations, self.scenarios[c]["constant_values"])]
                    df_scenario = dd.from_pandas(
                        pd.DataFrame(combinations, columns=scenario_columns + constant_columns),
                        npartitions=npartitions)
                    last_columns = [c for c in self.scenarios if
                                    self.scenarios[c]["mode"] == "last"]
                    if len(last_columns) > 0:
                        if self.target_dimensions:
                            last = df.groupby(self.target_dimensions)[last_columns].last().compute()
                            meta = df_scenario.join(last.reset_index(drop=True), how="right")

                            def join_func(target_dimension, time_index, df):
                                return df.set_index(target_dimension).join(
                                    last).reset_index().set_index(time_index).reset_index()

                            df_scenario = df_scenario.groupby(self.target_dimensions).apply(
                                partial(join_func, self.target_dimensions, self.time_index),
                                meta=meta).reset_index(drop=True)
                        else:
                            last = df[last_columns].tail(1)
                            for l in last_columns:
                                df_scenario[l] = last[l]

                    df = dd.concat([df.set_index(self.time_index),
                                    df_scenario.set_index(self.time_index)], axis=0).reset_index()

        if self.joins:
            for i, join in enumerate(self.joins):
                try:
                    if join["data_path"].startswith("divina://"):
                        join_df = _load(join["data_path"])
                    else:
                        join_df = dd.read_parquet("{}/*".format(join["data_path"]))
                except IndexError:
                    raise Exception("Could not load dataset {}. No parquet files found.".format(join["data_path"]))
                join_df[join["join_on"][0]] = join_df[join["join_on"][0]].astype(df[join["join_on"][1]].dtype)
                df = df.merge(
                    join_df,
                    how="left",
                    left_on=join["join_on"][0],
                    right_on=join["join_on"][1],
                    suffixes=("", "{}_".format(join["as"])),
                )

        for c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype(float)

        if self.include_features:
            df = df[[self.target, self.time_index] + self.include_features]

        if self.bin_features:
            for c in self.bin_features:
                edges = [-np.inf] + self.bin_features[c] + [np.inf]
                for v, v_1 in zip(edges, edges[1:]):
                    df["{}_({}, {}]".format(c, v, v_1)] = 1
                    df["{}_({}, {}]".format(c, v, v_1)] = df["{}_({}, {}]".format(c, v, v_1)].where(
                        ((df[c] < v_1) & (df[c] >= v)), 0)

        if self.encode_features:
            for c in self.encode_features:
                if df[c].dtype == int:
                    df[c] = df[c].astype(float)
                else:
                    df[c] = df[c]
                df["{}_dummy".format(c)] = df[c]

            pipe = make_pipeline(
                Categorizer(columns=self.encode_features),
                DummyEncoder(columns=self.encode_features))

            pipe.fit(df)

            df = pipe.transform(df)

            for c in self.encode_features:
                df[c] = df["{}_dummy".format(c)]
            df = df.drop(columns=["{}_dummy".format(c) for c in self.encode_features])

        if self.interaction_features:
            for t in self.interaction_features:
                if t in self.encode_features:
                    pipe = make_pipeline(
                        Categorizer(columns=[t]),
                        DummyEncoder(columns=[t]))
                    interactions = list(pipe.fit(df[[t]]).steps[1][1].transformed_columns_)
                else:
                    interactions = [t]
                for c in interactions:
                    for w in self.interaction_features[t]:
                        if w in self.encode_features:
                            pipe = make_pipeline(
                                Categorizer(columns=[w]),
                                DummyEncoder(columns=[w]))
                            v = list(pipe.fit(df[[w]]).steps[1][1].transformed_columns_)
                        else:
                            v = [w]
                        for m in v:
                            if not '{}-x-{}'.format(c, m) in df.columns:
                                if not all([is_numeric_dtype(x) for x in df[[t, m]].dtypes]):
                                    df['{}-x-{}'.format(c, m)] = df[t].astype(str) + "_*_" + df[m].astype(str)
                                else:
                                    df['{}-x-{}'.format(c, m)] = df[t] * df[m]

        if self.encode_features:
            df = df.drop(columns=self.encode_features)

        if self.drop_features:
            df = df.drop(columns=self.drop_features)

        df[self.time_index] = dd.to_datetime(df[self.time_index])
        df = df.repartition(npartitions=npartitions)
        df = cull_empty_partitions(df)
        df['index'] = 1
        df['index'] = df['index'].cumsum()
        df['index'] = df['index'] - 1
        df = df.set_index('index')
        df.index.name = None
        return df.copy().persist()

    def run(self, write_path, random_state=None):
        self.train(
            write_path=write_path,
            random_state=random_state
        )
        forecast = self.forecast(
            read_path=write_path,
            write_path=write_path
        )
        if self.validation_splits:
            self.validate(
                read_path=write_path,
                write_path=write_path
            )
        return forecast
