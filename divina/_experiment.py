import json
import os
import sys
from functools import partial
from itertools import product
from typing import Tuple, List
from .model import *

import backoff
import dask.array as da
import dask.dataframe as dd
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
import s3fs
from botocore.exceptions import ClientError
from dask_ml.linear_model import LinearRegression
from dask_ml.preprocessing import Categorizer, DummyEncoder
from pandas.api.types import is_numeric_dtype
from sklearn.pipeline import make_pipeline

from .datasets.load import _load
from .utils import create_write_directory, cull_empty_partitions, get_dask_client

from google.cloud import aiplatform
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Artifact
import pathlib
import numpy as np


class Validation():
    def __init__(self, metrics: dict, predictions: dd):
        self.metrics = metrics
        self.predictions = predictions


class CausalValidation(Validation):
    def __init__(self, bootstrap_validations: [Validation], metrics: dict, predictions: dd):
        self.bootstrap_validations = bootstrap_validations
        super().__init__(metrics, predictions)


class BoostValidation(Validation):
    def __init__(self, horizon: int, metrics: dict, predictions: dd):
        self.horizon = horizon
        super().__init__(metrics, predictions)


class ValidationSplit():
    def __init__(self, causal_validation: CausalValidation, split: str, boosted_validations: [BoostValidation],
                 truth: pd.DataFrame):
        self.causal_validation = causal_validation
        self.split = split
        self.truth = truth
        self.boosted_validations = boosted_validations


class ExperimentResult():
    def __init__(self, split_validations: [ValidationSplit]):
        self.split_validations = split_validations


class Experiment:
    def __init__(
            self,
            target,
            time_index,
            target_dimensions=None,
            include_features=None,
            drop_features=None,
            joins=None,
            encode_features=None,
            bin_features=None,
            interaction_features=None,
            time_horizons=None,
            train_start=None,
            train_end=None,
            forecast_start=None,
            forecast_end=None,
            validate_start=None,
            validate_end=None,
            validation_splits=None,
            link_function=None,
            confidence_intervals=None,
            random_seed=None,
            bootstrap_sample=None,
            scenarios=None,
            frequency=None,
    ):
        if not time_horizons:
            self.time_horizons = [0]
        else:
            horizon_ranges = [x for x in time_horizons if type(x) == tuple]
            if len(horizon_ranges) > 0:
                self.time_horizons = [x for x in time_horizons if type(x) == int]
                for x in horizon_ranges:
                    self.time_horizons = set(
                        self.time_horizons + list(range(x[0], x[1]))
                    )
            self.time_horizons = time_horizons
        if not confidence_intervals:
            self.confidence_intervals = []
        else:
            self.confidence_intervals = confidence_intervals
        self._bootstrap_sample = 0
        if not bootstrap_sample:
            self.bootstrap_sample = 0
        else:
            self.bootstrap_sample = bootstrap_sample
        self.target_dimensions = target_dimensions
        self.joins = joins
        self.include_features = include_features
        self.drop_features = drop_features
        self.encode_features = encode_features
        self.bin_features = bin_features
        self.interaction_features = interaction_features
        self.train_start = train_start
        self.train_end = train_end
        self.forecast_start = forecast_start
        self.forecast_end = forecast_end
        self.validate_start = validate_start
        self.validate_end = validate_end
        self.validation_splits = validation_splits
        self.link_function = link_function
        self.target = target
        self.time_index = time_index
        self.random_seed = random_seed
        self.scenarios = scenarios
        self.frequency = frequency
        self.models = {}
        self.metrics = {}

    @property
    def bootstrap_sample(self):
        return self._bootstrap_sample

    @bootstrap_sample.setter
    def bootstrap_sample(self, value):
        self._bootstrap_sample = value
        if hasattr(self, 'random_seed') and self.random_seed:
            self.bootstrap_seeds = [
                x for x in range(self.random_seed, self.random_seed + value)
            ]
        else:
            self.bootstrap_seeds = [x for x in np.random.randint(0, 10000, size=value)]
        return

    @staticmethod
    def _preprocess(target: str, scenarios: dict, encode_features: list, frequency: str,
                    include_features: List, drop_features: List, bin_features: List, interaction_features: dict,
                    time_index: str, target_dimensions: List, dataframe: dd = None, data_uri: str = None,
                    dataset_uri: str = None, start=None, end=None):

        if type(dataframe) == type(None) and type(data_uri) == type(None):
            raise ValueError('Either dataframe or data_uri arguments must be specified')
        if not type(dataframe) == type(None) and not type(data_uri) == type(None):
            raise ValueError('Only one of either dataframe or data_uri arguments may be specified')
        if data_uri:
            if data_uri.startswith("divina://"):
                df = _load(data_uri)
            else:
                df = dd.read_parquet("{}/*".format(data_uri))
        else:
            df = dataframe
        npartitions = (df.memory_usage(deep=True).sum().compute() // 104857600) + 1

        df[time_index] = dd.to_datetime(df[time_index])

        time_min, time_max = (
            df[time_index].min().compute(),
            df[time_index].max().compute(),
        )

        if target_dimensions:
            df = (
                df.groupby([time_index] + target_dimensions)
                    .agg(
                    {
                        **{
                            c: "sum"
                            for c in df.columns
                            if df[c].dtype in [int, float] and c != time_index
                        },
                        **{
                            c: "first"
                            for c in df.columns
                            if df[c].dtype not in [int, float] and c != time_index
                        },
                    }
                ).drop(columns=target_dimensions).reset_index()
            )
        else:
            df = (
                df.groupby(time_index)
                    .agg(
                    {
                        **{
                            c: "sum"
                            for c in df.columns
                            if df[c].dtype in [int, float] and c != time_index
                        },
                        **{
                            c: "first"
                            for c in df.columns
                            if df[c].dtype not in [int, float] and c != time_index
                        },
                    }
                )
                    .reset_index()
            )

        if start:
            if pd.to_datetime(start) < time_min:
                raise Exception(
                    "Bad Start: {} < {} Check Dataset Time Range".format(
                        start, time_min
                    )
                )
            else:
                df = df[dd.to_datetime(df[time_index]) >= start]

        if end:
            if pd.to_datetime(end) > time_max:
                if not scenarios:
                    raise Exception(
                        "Bad End: {} | {} Check Dataset Time Range".format(
                            end, time_max
                        )
                    )
            else:
                df = df[dd.to_datetime(df[time_index]) <= end]

        if scenarios:
            if not frequency:
                raise Exception(
                    'Frequency of time series must be supplied. Please supply with "frequency: "D", "M", "s", etc."'
                )
            if end:
                if start and pd.to_datetime(start) > time_max:
                    new_dates = pd.date_range(
                        pd.to_datetime(str(start)),
                        pd.to_datetime(str(end)),
                        freq=frequency,
                    )
                else:
                    new_dates = pd.date_range(
                        time_max + pd.tseries.frequencies.to_offset(frequency),
                        pd.to_datetime(str(end)),
                        freq=frequency,
                    )
                if len(new_dates) > 0:

                    combinations = list(new_dates)
                    if target_dimensions:
                        combinations = [
                            list(x)
                            for x in product(
                                combinations,
                                *[
                                    df[s].unique().compute().values
                                    for s in target_dimensions
                                ]
                            )
                        ]
                        scenario_columns = [time_index] + target_dimensions
                    else:
                        combinations = [[x] for x in combinations]
                        scenario_columns = [time_index]
                    constant_columns = [
                        c
                        for c in scenarios
                        if scenarios[c]["mode"] == "constant"
                    ]
                    for c in constant_columns:
                        combinations = [
                            x[0] + [x[1]]
                            for x in product(
                                combinations, scenarios[c]["constant_values"]
                            )
                        ]
                    df_scenario = dd.from_pandas(
                        pd.DataFrame(
                            combinations, columns=scenario_columns + constant_columns
                        ),
                        npartitions=npartitions,
                    )
                    last_columns = [
                        c for c in scenarios if scenarios[c]["mode"] == "last"
                    ]
                    if len(last_columns) > 0:
                        if target_dimensions:
                            last = (
                                df.groupby(target_dimensions)[last_columns]
                                    .last()
                                    .compute()
                            )
                            meta = df_scenario.join(
                                last.reset_index(drop=True), how="right"
                            )

                            def join_func(target_dimension, time_index, df):
                                return (
                                    df.set_index(target_dimension)
                                        .join(last)
                                        .reset_index()
                                        .set_index(time_index)
                                        .reset_index()
                                )

                            df_scenario = (
                                df_scenario.groupby(target_dimensions)
                                    .apply(
                                    partial(
                                        join_func,
                                        target_dimensions,
                                        time_index,
                                    ),
                                    meta=meta,
                                )
                                    .reset_index(drop=True)
                            )
                        else:
                            last = df[last_columns].tail(1)
                            for l in last_columns:
                                df_scenario[l] = last[l]

                    df = dd.concat(
                        [
                            df.set_index(time_index),
                            df_scenario.set_index(time_index),
                        ],
                        axis=0,
                    ).reset_index()

        for c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype(float)

        if include_features:
            df = df[[target, time_index] + include_features]

        if bin_features:
            for c in bin_features:
                edges = [-np.inf] + bin_features[c] + [np.inf]
                for v, v_1 in zip(edges, edges[1:]):
                    df["{}_({}, {}]".format(c, v, v_1)] = 1
                    df["{}_({}, {}]".format(c, v, v_1)] = df[
                        "{}_({}, {}]".format(c, v, v_1)
                    ].where(((df[c] < v_1) & (df[c] >= v)), 0)

        if encode_features:
            for c in encode_features:
                if df[c].dtype == int:
                    df[c] = df[c].astype(float)
                else:
                    df[c] = df[c]
                df["{}_dummy".format(c)] = df[c]

            pipe = make_pipeline(
                Categorizer(columns=encode_features),
                DummyEncoder(columns=encode_features),
            )

            pipe.fit(df)

            df = pipe.transform(df)

            for c in encode_features:
                df[c] = df["{}_dummy".format(c)]
            df = df.drop(columns=["{}_dummy".format(c) for c in encode_features])

        if interaction_features:
            for t in interaction_features:
                if t in encode_features:
                    pipe = make_pipeline(
                        Categorizer(columns=[t]), DummyEncoder(columns=[t])
                    )
                    interactions = list(
                        pipe.fit(df[[t]]).steps[1][1].transformed_columns_
                    )
                else:
                    interactions = [t]
                for c in interactions:
                    for w in interaction_features[t]:
                        if w in encode_features:
                            pipe = make_pipeline(
                                Categorizer(columns=[w]), DummyEncoder(columns=[w])
                            )
                            v = list(pipe.fit(df[[w]]).steps[1][1].transformed_columns_)
                        else:
                            v = [w]
                        for m in v:
                            if not "{}-x-{}".format(c, m) in df.columns:
                                if not all(
                                        [is_numeric_dtype(x) for x in df[[t, m]].dtypes]
                                ):
                                    df["{}-x-{}".format(c, m)] = (
                                            df[t].astype(str) + "_*_" + df[m].astype(str)
                                    )
                                else:
                                    df["{}-x-{}".format(c, m)] = df[t] * df[m]

        if encode_features:
            df = df.drop(columns=encode_features)

        if drop_features:
            df = df.drop(columns=drop_features)

        df_std = df[[c for c in df.columns if not c == target]].std().compute()
        constant_columns = [c for c in df_std.index if df_std[c] == 0]
        df = df.drop(columns=constant_columns)

        df[time_index] = dd.to_datetime(df[time_index])
        df = df.repartition(npartitions=npartitions)
        df = cull_empty_partitions(df)
        if target_dimensions:
            df["__target_dimension_index__"] = df[time_index].astype(str)
            for i, col in enumerate(target_dimensions):
                df["__target_dimension_index__"] += "__index__".format(i) + df[col].astype(str)
            df = df.drop(columns=[time_index] + target_dimensions)
            df = df.set_index("__target_dimension_index__")
        else:
            df = df.set_index(time_index)
        if dataset_uri:
            dd.to_parquet(dataset_uri)
        return df.copy().persist()

    def preprocess(self, dataframe: dd = None, data_uri: str = None, start=None, end=None, kfp=False):
        @component(packages_to_install=['dask[dataframe]', 'pyarrow', 'gcsfs', 'dill'],
                   base_image='us-docker.pkg.dev/python:3.9')
        @create_write_directory
        @get_dask_client
        @backoff.on_exception(backoff.expo, ClientError, max_time=30)
        def _preprocess_component(data_uri: str, dataset: Output[Dataset],
                                  target: str, scenarios: dict, encode_features: list, frequency: str,
                                  include_features: List, drop_features: List, bin_features: List,
                                  interaction_features: dict, time_index: str, target_dimensions: List, start=start,
                                  end=end):
            ###TODO override dataset component with supplied uri
            return Experiment._preprocess(data_uri=data_uri, dataset_uri=dataset.uri,
                                          target=target, scenarios=scenarios,
                                          encode_features=encode_features, frequency=frequency,
                                          include_features=include_features, drop_features=drop_features,
                                          bin_features=bin_features, interaction_features=interaction_features,
                                          time_index=time_index, target_dimensions=target_dimensions, start=start,
                                          end=end)

        if kfp:
            return _preprocess_component(data_uri=data_uri,
                                         target=self.target, scenarios=self.scenarios, joins=self.joins,
                                         encode_features=self.encode_features, frequency=self.frequency,
                                         include_features=self.include_features, drop_features=self.drop_features,
                                         bin_features=self.bin_features,
                                         interaction_features=self.interaction_features, time_index=self.time_index,
                                         target_dimensions=self.target_dimensions, start=start, end=end)
        else:
            return Experiment._preprocess(data_uri=data_uri, dataframe=dataframe, target=self.target,
                                          scenarios=self.scenarios,
                                          encode_features=self.encode_features,
                                          frequency=self.frequency, include_features=self.include_features,
                                          drop_features=self.drop_features, bin_features=self.bin_features,
                                          interaction_features=self.interaction_features, time_index=self.time_index,
                                          target_dimensions=self.target_dimensions,
                                          start=start, end=end)

    @staticmethod
    def _train(model_type: str, model_params: dict, random_state: int = 11,
               X: dd = None,
               X_uri: str = None,
               y: dd = None,
               y_uri: str = None,
               model_uri: str = None, params_uri: str = None,
               bootstrap: bool = False, horizon: int = 0):

        if type(X) == type(None) and type(X_uri) == type(None):
            raise ValueError('either dataset or dataset_uri must be specified')
        if type(X) != type(None) and type(X_uri) != type(None):
            raise ValueError('only one of either dataset or dataset_uri can be specified')
        if type(y) == type(None) and type(y_uri) == type(None):
            raise ValueError('either dataset or dataset_uri must be specified')
        if type(y) != type(None) and type(y_uri) != type(None):
            raise ValueError('only one of either dataset or dataset_uri can be specified')

        if X_uri:
            X = dd.read_parquet(X_uri)
        if y_uri:
            y = dd.read_parquet(y_uri)

        if y.isna().sum().compute()[0] > 0:
            raise ValueError('Null values in target not permitted.')

        if bootstrap:
            X = X.sample(
                replace=False, frac=0.8, random_state=random_state
            )
            y = y.sample(
                replace=False, frac=0.8, random_state=random_state
            )

        if model_params:
            model = eval(model_type)(**model_params)
        else:
            model = eval(model_type)()

        model.fit(
            X.to_dask_array(lengths=True),
            y.shift(-horizon).dropna().to_dask_array(lengths=True),
        )
        if model_uri:
            with open(
                    model_uri,
                    "wb",
            ) as f:
                joblib.dump(model, f)
        params = {"model_type": model_type, "model_params": model_params}
        if params_uri:
            with open(
                    params_uri,
                    "w",
            ) as f:
                json.dump(params, f)

        return model

    def train(self, model_type, model_params=None, X_dataset=None, X=None, y_dataset=None, y=None, random_state=None,
              bootstrap=False, horizon=0,
              kfp=False):
        @component(packages_to_install=['dask[dataframe]', 'pyarrow', 'gcsfs', 'dill'],
                   base_image='us-docker.pkg.dev/python:3.9')
        @create_write_directory
        @get_dask_client
        @backoff.on_exception(backoff.expo, ClientError, max_time=30)
        def _train_component(X_dataset: Input[Dataset], y_dataset: Input[Dataset], model: Output[Model],
                             params: Output[Artifact],
                             random_state,
                             bootstrap,
                             model_type, model_params, horizon=horizon):
            Experiment._train(X_uri=X_dataset.uri, y_uri=y_dataset.uri, model_uri=model.uri, params_uri=params.uri,
                              random_state=random_state,
                              bootstrap=bootstrap,
                              model_type=model_type, model_params=model_params, horizon=horizon)

        if kfp:
            return _train_component(X_dataset=X_dataset,
                                    y_dataset=y_dataset,
                                    random_state=random_state,
                                    bootstrap=bootstrap,
                                    model_type=model_type, model_params=model_params, horizon=horizon)
        else:
            return Experiment._train(X=X, y=y, random_state=random_state,
                                     bootstrap=bootstrap,
                                     model_type=model_type, model_params=model_params, horizon=horizon)

    def forecast(self, X_dataset=None, model=None, X=None, kfp=False):
        @component(packages_to_install=['dask[dataframe]', 'pyarrow', 'gcsfs', 'dill'],
                   base_image='us-docker.pkg.dev/python:3.9')
        @create_write_directory
        @get_dask_client
        @backoff.on_exception(backoff.expo, ClientError, max_time=30)
        def _forecast_component(X_dataset: Input[Dataset], model: Input[Model], predictions: Output[Dataset]):
            Experiment._forecast(X_uri=X_dataset.uri, model_uri=model.uri, predictions_uri=predictions.uri,
                                 )

        if kfp:
            return _forecast_component(X_dataset=X_dataset, model=model,
                                       )
        else:
            return Experiment._forecast(X=X, model=model,
                                        )

    @staticmethod
    def _forecast(model_uri=None, model=None, X: dd = None, X_uri=None,
                  predictions_uri=None):

        if type(X) == type(None) and type(X_uri) == type(None):
            raise ValueError('either dataset or dataset_uri must be specified')
        if type(X) != type(None) and type(X_uri) != type(None):
            raise ValueError('only one of either dataset or dataset_uri can be specified')
        if type(model) == type(None) and type(model_uri) == type(None):
            raise ValueError('either model or model_uri must be specified')
        if type(model) != type(None) and type(model_uri) != type(None):
            raise ValueError('only one of either model or model_uri can be specified')

        if X_uri:
            X = dd.read_parquet(X_uri)

        if model_uri:
            with open(model_uri, 'rb') as f:
                model = joblib.load(f)

            ###TODO put this in preprocessing

        features = X.columns
        for f in features:
            if not is_numeric_dtype(X[f].dtype):
                try:
                    X[f] = X[f].astype(float)
                except ValueError:
                    raise ValueError(
                        "{} could not be converted to float. "
                        "Please convert to numeric or encode with "
                        '"encode_features: {}"'.format(f, f)
                    )

        y_hat = dd.from_dask_array(model.predict(
            X.to_dask_array(lengths=True)
        )).to_frame()
        y_hat.index = X.index

        '''
        factor_df = dd.from_dask_array(
            df[features].to_dask_array(lengths=True)
            * da.from_array(model.linear_model.coef_)
        )
        factor_df.columns = ["factor_{}".format(c) for c in features]
        for c in factor_df:
            df[c] = factor_df[c]
        '''
        if predictions_uri:
            dd.to_parquet(
                y_hat,
                predictions_uri,
            )

        return y_hat

    def validate(self, truth_dataset=None, prediction_dataset=None, kfp=False):
        @component(packages_to_install=['dask[dataframe]', 'pyarrow', 'gcsfs', 'dill'],
                   base_image='us-docker.pkg.dev/python:3.9')
        @create_write_directory
        @get_dask_client
        @backoff.on_exception(backoff.expo, ClientError, max_time=30)
        def _validate_component(validation: Output[Artifact], truth_dataset: Input[Dataset],
                                prediction_dataset: Input[Dataset]):
            Experiment._validate(df_truth_uri=truth_dataset.uri, df_predictions_uri=prediction_dataset.uri,
                                 target=self.target, validation_uri=validation.uri)

        if kfp:
            return _validate_component(truth_dataset=truth_dataset, prediction_dataset=prediction_dataset,
                                       link_function=self.link_function)
        else:
            return Experiment._validate(df_truth=truth_dataset, df_predictions=prediction_dataset,
                                        target=self.target)

    @staticmethod
    def _validate(target: str, validation_uri: str = None, df_truth: dd = None, df_predictions: dd = None,
                  df_truth_uri: str = None, df_predictions_uri: str = None):

        if type(df_truth) == type(None) and type(df_truth_uri) == type(None):
            raise ValueError('either df_truth or df_truth_uri must be specified')
        if type(df_truth) != type(None) and type(df_truth_uri) != type(None):
            raise ValueError('only one of either df_truth or df_truth_uri can be specified')
        if type(df_predictions) == type(None) and type(df_predictions_uri) == type(None):
            raise ValueError('either df_prediction or df_prediction_uri must be specified')
        if type(df_predictions) != type(None) and type(df_predictions_uri) != type(None):
            raise ValueError('only one of either df_prediction or df_prediction_uri can be specified')

        if df_truth_uri:
            df_truth = dd.read_parquet(df_truth_uri)

        if df_predictions_uri:
            df_predictions = dd.read_parquet(df_predictions_uri)

        residuals = df_predictions[df_predictions.columns[0]] - df_truth[target]

        metrics = {"mse": residuals.pow(2).mean().compute()}

        if validation_uri:
            with open(validation_uri, 'r') as f:
                json.dump(metrics, f)

        return metrics

    def split_dataset(self, dataset, split, kfp=False):
        @component(packages_to_install=['dask[dataframe]', 'pyarrow', 'gcsfs', 'dill'],
                   base_image='us-docker.pkg.dev/python:3.9')
        @create_write_directory
        @get_dask_client
        @backoff.on_exception(backoff.expo, ClientError, max_time=30)
        def _split_dataset_component(split: str, dataset: Input[Dataset], train_df: Output[Dataset],
                                     test_df: Output[Dataset]):
            Experiment._split_dataset(split, dataset_uri=dataset.uri, df_train_uri=train_df.uri,
                                      df_test_uri=test_df.uri)

        if kfp:
            return _split_dataset_component(split, dataset=dataset)
        else:
            return Experiment._split_dataset(split, df=dataset)

    @staticmethod
    def _split_dataset(split, df: dd = None, dataset_uri: str = None, df_train_uri=None, df_test_uri=None):

        if type(df) == type(None) and type(dataset_uri) == type(None):
            raise ValueError('either dataset or dataset_uri must be specified')
        if type(df) != type(None) and type(dataset_uri) != type(None):
            raise ValueError('only one of either dataset or dataset_uri can be specified')

        if dataset_uri:
            df = dd.read_parquet(dataset_uri)

        npartitions = (df.memory_usage(deep=True).sum().compute() // 104857600) + 1

        if df.index.name == '__target_dimension_index__':
            df_time = df.reset_index()['__target_dimension_index__'].str.split('__index__', expand=True, n=len(df.head(1).index[0].split('__index__'))-1).set_index(0)
        else:
            df_time = df

        train_df = df.loc[dd.to_datetime(df_time.index) < split]
        test_df = df.loc[dd.to_datetime(df_time.index) >= split]

        train_df = train_df.repartition(npartitions=npartitions)
        train_df = cull_empty_partitions(train_df)
        test_df = test_df.repartition(npartitions=npartitions)
        test_df = cull_empty_partitions(test_df)

        if df_train_uri:
            dd.to_parquet(
                train_df,
                df_train_uri,
            )
        if df_test_uri:
            dd.to_parquet(
                train_df,
                df_train_uri,
            )

        return train_df, test_df

    def x_y_split(self, dataset, target, kfp=False):
        @component(packages_to_install=['dask[dataframe]', 'pyarrow', 'gcsfs', 'dill'],
                   base_image='us-docker.pkg.dev/python:3.9')
        @create_write_directory
        @get_dask_client
        @backoff.on_exception(backoff.expo, ClientError, max_time=30)
        def _x_y_split_component(target: str, dataset: Input[Dataset], X: Output[Dataset],
                                 y: Output[Dataset]):
            Experiment._x_y_split(target, dataset_uri=dataset.uri, X_uri=X.uri,
                                  y_uri=y.uri)

        if kfp:
            return _x_y_split_component(target, dataset=dataset)
        else:
            return Experiment._x_y_split(target, df=dataset)

    @staticmethod
    def _x_y_split(target, df: dd = None, dataset_uri: str = None, X_uri=None, y_uri=None):

        if type(df) == type(None) and type(dataset_uri) == type(None):
            raise ValueError('either dataset or dataset_uri must be specified')
        if type(df) != type(None) and type(dataset_uri) != type(None):
            raise ValueError('only one of either dataset or dataset_uri can be specified')

        if dataset_uri:
            df = dd.read_parquet(dataset_uri)

        X = df[[c for c in df.columns if not c == target]]
        y = df[[target]]

        if X_uri:
            dd.to_parquet(
                X,
                X_uri,
            )
        if y_uri:
            dd.to_parquet(
                y,
                y_uri,
            )

        return X, y

    ##TODO - https://stackoverflow.com/questions/59445167/kubeflow-pipeline-dynamic-output-list-as-input-parameter

    def aggregate_forecasts(self, forecasts, kfp=False):
        @component(packages_to_install=['dask[dataframe]', 'pyarrow', 'gcsfs', 'dill'],
                   base_image='us-docker.pkg.dev/python:3.9')
        @create_write_directory
        @get_dask_client
        @backoff.on_exception(backoff.expo, ClientError, max_time=30)
        def _aggregate_forecasts_component(forecasts: [Input[Dataset]], interval: Output[Dataset],
                                           point_estimate: Output[Dataset]):
            Experiment._aggregate_forecasts(forecast_uris=[d.uri for d in forecasts], interval_uri=interval.uri,
                                            point_estimate_uri=point_estimate.uri,
                                            confidence_intervals=self.confidence_intervals, target=self.target)

        if kfp:
            return _aggregate_forecasts_component(forecasts=forecasts)
        else:
            return Experiment._aggregate_forecasts(forecasts=forecasts, confidence_intervals=self.confidence_intervals,
                                                   target=self.target)

    @staticmethod
    def _aggregate_forecasts(target, confidence_intervals, forecasts: [dd] = None, forecast_uris: [str] = None,
                             interval_uri=None, point_estimate_uri=None):

        if type(forecasts) == type(None) and type(forecast_uris) == type(None):
            raise ValueError('either dataset or dataset_uri must be specified')
        if type(forecasts) != type(None) and type(forecast_uris) != type(None):
            raise ValueError('only one of either dataset or dataset_uri can be specified')

        if forecast_uris:
            forecasts = [dd.read_parquet(uri) for uri in forecast_uris]

        df_forecasts = dd.concat([dd.from_dask_array(s.to_dask_array(lengths=True)) for s in forecasts],
                                 axis=1)
        df_forecasts.columns = ['bootstrap_{}'.format(c) for c in range(len(forecasts))]
        df_interval = dd.from_dask_array(dd.from_dask_array(df_forecasts.to_dask_array(lengths=True).T).quantile(
            [i * 0.01 for i in confidence_intervals]).to_dask_array(lengths=True).T)

        df_interval.columns = [
            "{}_pred_c_{}".format(target, c)
            for c in confidence_intervals
        ]

        df_point_estimate = dd.from_dask_array(
            dd.from_dask_array(df_forecasts.to_dask_array(lengths=True).T).mean().to_dask_array(
                lengths=True).T).to_frame()

        df_point_estimate.columns = ['mean']

        df_interval = df_interval.repartition(
            divisions=df_forecasts.divisions
        )

        df_interval.index = forecasts[0].index

        df_point_estimate = df_point_estimate.repartition(
            divisions=df_forecasts.divisions
        )

        df_point_estimate.index = forecasts[0].index

        if interval_uri:
            dd.to_parquet(
                df_interval,
                interval_uri,
            )

        if point_estimate_uri:
            dd.to_parquet(
                df_point_estimate,
                point_estimate_uri,
            )

        return df_interval, df_point_estimate

    def long_to_wide_residuals(self, prediction_series, truth_series, horizon, window, target_dimensions, kfp=False):
        @component(packages_to_install=['dask[dataframe]', 'pyarrow', 'gcsfs', 'dill'],
                   base_image='us-docker.pkg.dev/python:3.9')
        @create_write_directory
        @get_dask_client
        @backoff.on_exception(backoff.expo, ClientError, max_time=30)
        def _long_to_wide_residuals(prediction_series: Input[Dataset], truth_series: Input[Dataset],
                                    wide_format: Output[Dataset]):
            Experiment._long_to_wide_residuals(prediction_series_uri=prediction_series.uri,
                                               truth_series_uri=truth_series.uri, wide_format_uri=wide_format.uri,
                                               horizon=horizon, window=window, target_dimensions=target_dimensions)

        if kfp:
            return _long_to_wide_residuals(prediction_series=prediction_series, truth_series=truth_series,
                                           horizon=horizon, window=window, target_dimensions=target_dimensions)
        else:
            return Experiment._long_to_wide_residuals(prediction_series=prediction_series, truth_series=truth_series,
                                                      horizon=horizon, window=window,
                                                      target_dimensions=target_dimensions)

    @staticmethod
    def _long_to_wide_residuals(prediction_series: dd = None, prediction_series_uri: str = None,
                                truth_series: dd = None, truth_series_uri: str = None, wide_format_uri: str = None,
                                horizon: int = 0, window: int = 0, target_dimensions=None):

        if type(prediction_series) == type(None) and type(prediction_series_uri) == type(None):
            raise ValueError('either dataset or dataset_uri must be specified')
        if type(prediction_series) != type(None) and type(prediction_series_uri) != type(None):
            raise ValueError('only one of either dataset or dataset_uri can be specified')

        if type(truth_series) == type(None) and type(truth_series_uri) == type(None):
            raise ValueError('either dataset or dataset_uri must be specified')
        if type(truth_series) != type(None) and type(truth_series_uri) != type(None):
            raise ValueError('only one of either dataset or dataset_uri can be specified')

        if len(prediction_series) < horizon + window:
            raise ValueError('not enough observations to convert to wide format given window and horizon')

        if prediction_series_uri:
            prediction_series = dd.read_parquet(prediction_series_uri)

        if truth_series_uri:
            truth_series = dd.read_parquet(truth_series_uri)

        series_name = prediction_series.columns[0]

        residual_series = (prediction_series[series_name] - truth_series[truth_series.columns[0]]).to_frame()

        residual_series.columns = [series_name]

        lags = range(horizon, horizon + window)

        def _create_lags(series_name, lags, df):
            for lag in lags:
                df['lag_{}'.format(lag)] = df[series_name].shift(-lag)
            return df

        if target_dimensions:
            target_dimensions_df = residual_series.reset_index()['__target_dimension_index__'].str.split('__index__', expand=True, n=len(residual_series.head(1).index[0].split('__index__'))-1)
            target_dimensions_df.index = residual_series.index
            target_dimensions_df.columns = ['time'] + target_dimensions
            for c in target_dimensions:
                residual_series[c] = target_dimensions_df[c]
            residual_series = residual_series.groupby(target_dimensions).apply(
                partial(_create_lags, series_name, lags)).drop(columns=target_dimensions)

        else:
            residual_series = _create_lags(series_name, lags, residual_series)

        residual_series.index = prediction_series.index

        if wide_format_uri:
            dd.to_parquet(
                residual_series,
                wide_format_uri,
            )

        return residual_series

    def boost_forecast(self, forecast_series, adjustment_series, kfp=False):
        @component(packages_to_install=['dask[dataframe]', 'pyarrow', 'gcsfs', 'dill'],
                   base_image='us-docker.pkg.dev/python:3.9')
        @create_write_directory
        @get_dask_client
        @backoff.on_exception(backoff.expo, ClientError, max_time=30)
        def _boost_forecast(forecast_series: Input[Dataset], adjustment_series: Input[Dataset],
                            boosted_forecast: Output[Dataset]):
            Experiment._boost_forecast(forecast_series_uri=forecast_series.uri,
                                       adjustment_series_uri=adjustment_series.uri,
                                       boosted_forecast_uri=boosted_forecast.uri,
                                       )

        if kfp:
            return _boost_forecast(forecast_series_uri=forecast_series.uri,
                                   adjustment_series_uri=adjustment_series.uri)
        else:
            return Experiment._boost_forecast(forecast_series=forecast_series,
                                              adjustment_series=adjustment_series)

    @staticmethod
    def _boost_forecast(forecast_series: dd = None, forecast_series_uri: str = None,
                        adjustment_series: dd = None, adjustment_series_uri: str = None,
                        boosted_forecast_uri: str = None):

        if type(forecast_series) == type(None) and type(forecast_series_uri) == type(None):
            raise ValueError('either dataset or dataset_uri must be specified')
        if type(forecast_series) != type(None) and type(forecast_series_uri) != type(None):
            raise ValueError('only one of either dataset or dataset_uri can be specified')

        if type(adjustment_series) == type(None) and type(adjustment_series_uri) == type(None):
            raise ValueError('either dataset or dataset_uri must be specified')
        if type(adjustment_series) != type(None) and type(adjustment_series_uri) != type(None):
            raise ValueError('only one of either dataset or dataset_uri can be specified')

        if forecast_series_uri:
            forecast_series = dd.read_parquet(forecast_series_uri)

        if adjustment_series_uri:
            adjustment_series = dd.read_parquet(adjustment_series_uri)

        boosted_forecast = forecast_series + adjustment_series

        if boosted_forecast_uri:
            dd.to_parquet(
                boosted_forecast,
                boosted_forecast_uri,
            )

        return boosted_forecast

    def run(self, data, causal_model_type='GLM', boost_model_type='EWMA', causal_model_params=None,
            boost_model_params=None, kfp=False):
        @dsl.pipeline(
            name="test-pipeline",
            description="testing",
            pipeline_root='gs://arbiter-datasets-raw/test-pipeline',
        )
        def _run_pipeline_kfp():
            df = self.preprocess(data_uri=data.uri, kfp=kfp)
            validation_splits = []

            for s in self.validation_splits:
                df_splits = self.split_dataset(df.outputs['df'], s, kfp=kfp)
                bootstrap_predictions = []
                bootstrap_validations = []
                boosted_validations = []
                ###TODO add X and y split component here 0
                for n in self.bootstrap_seeds:
                    bootstrap_model = self.train(causal_model_type, causal_model_params, df_splits.outputs['train_df'],
                                                 bootstrap=True, random_state=n,
                                                 kfp=kfp)
                    bootstrap_prediction = self.forecast(df_splits.outputs['test_df'],
                                                         bootstrap_model.outputs['model'], kfp=kfp)
                    bootstrap_validation = self.validate(df_splits.outputs['test_df'],
                                                         bootstrap_prediction.outputs['predictions'], kfp=kfp)
                    bootstrap_predictions.append(bootstrap_prediction.outputs['predictions'])
                    bootstrap_validations.append(bootstrap_validation.outputs['validation'])
                causal_predictions = self.aggregate_forecasts(bootstrap_predictions)
                causal_validation = self.validate(df_splits.outputs['test_df'],
                                                  causal_predictions.output['predictions'], kfp=kfp)
                for h in self.time_horizons:
                    boosted_model = self.train(boost_model_type, boost_model_params, df_splits.outputs['train_df'],
                                               causal_predictions.output['predictions'], h, kfp=kfp)
                    boosted_predictions = self.forecast(df_splits.outputs['test_df'],
                                                        causal_predictions.output['predictions'],
                                                        boosted_model.outputs['model'], kfp=kfp)
                    boosted_validation = self.validate(df_splits.outputs['test_df'],
                                                       boosted_predictions.output['predictions'], kfp=kfp)
                    boosted_validations.append(boosted_validation.outputs['validation'])
                validation_splits.append(ValidationSplit(causal_validation=causal_validation.outputs['validation'],
                                                         boosted_validations=boosted_validations, split=s,
                                                         truth=df_splits.outputs['test_df']))
            return ExperimentResult(split_validations=validation_splits)

        def _run_pipeline(df, causal_model_type, boost_model_type, causal_model_params=None,
                          boost_model_params=None):

            df = self.preprocess(df)
            validation_splits = []

            for s in self.validation_splits:
                train_df, test_df = self.split_dataset(df, s)
                bootstrap_predictions = []
                bootstrap_validations = []
                boosted_validations = []
                X_train, y_train = self.x_y_split(train_df, self.target)
                X_test, y_test = self.x_y_split(test_df, self.target)
                for n in self.bootstrap_seeds:
                    bootstrap_model = self.train(model_type=causal_model_type, model_params=causal_model_params,
                                                 X=X_train,
                                                 y=y_train, bootstrap=True,
                                                 random_state=n)
                    bootstrap_prediction = self.forecast(X=X_test, model=bootstrap_model)
                    bootstrap_validation = self.validate(truth_dataset=y_test, prediction_dataset=bootstrap_prediction)
                    bootstrap_predictions.append(bootstrap_prediction)
                    bootstrap_validations.append(bootstrap_validation)
                confidence_intervals, point_estimates = self.aggregate_forecasts(bootstrap_predictions)
                causal_validation = self.validate(truth_dataset=y_test, prediction_dataset=point_estimates)
                for h in self.time_horizons:
                    wide_residuals = self.long_to_wide_residuals(prediction_series=point_estimates, truth_series=y_test,
                                                                 horizon=h, window=boost_model_params['window'],
                                                                 target_dimensions=self.target_dimensions)
                    X, y = self.x_y_split(wide_residuals, 'mean')
                    boosted_model = self.train(model_type=boost_model_type, model_params=boost_model_params, X=X, y=y)
                    residual_predictions = self.forecast(X=X, model=boosted_model)
                    boosted_predictions = self.boost_forecast(forecast_series=point_estimates,
                                                              adjustment_series=residual_predictions)
                    boosted_validation = self.validate(prediction_dataset=boosted_predictions, truth_dataset=y_test)
                    boosted_validations.append(boosted_validation)
                validation_splits.append(
                    ValidationSplit(causal_validation=causal_validation, boosted_validations=boosted_validations,
                                    split=s, truth=test_df))
            return ExperimentResult(split_validations=validation_splits)

        if kfp:
            return _run_pipeline_kfp(data, causal_model_type=causal_model_type, boost_model_type=boost_model_type,
                                     causal_model_params=causal_model_params,
                                     boost_model_params=boost_model_params)
        else:
            return _run_pipeline(data, causal_model_type=causal_model_type, boost_model_type=boost_model_type,
                                 causal_model_params=causal_model_params,
                                 boost_model_params=boost_model_params)
