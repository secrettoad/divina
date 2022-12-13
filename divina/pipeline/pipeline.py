from .model import *
from typing import Union
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.pipeline import make_pipeline
from pipeline.utils import _divina_component
from pipeline.utils import Output
import dask.dataframe as dd
from dask_ml.preprocessing import Categorizer, DummyEncoder
import numpy as np
from dask_ml.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from itertools import product


class Validation():
    def __init__(self, metrics: dict, predictions: dd, model: BaseEstimator = None):
        self.metrics = metrics
        self.predictions = predictions
        self.model = model

    def __eq__(self, other):
        return self.metrics == other.metrics and self.predictions.compute().values == other.predictions.compute().values and self.model == other.model


class CausalValidation(Validation):
    def __init__(self, bootstrap_validations: [Validation], metrics: dict, predictions: dd):
        self.bootstrap_validations = bootstrap_validations
        super().__init__(metrics, predictions)

    def __eq__(self, other):
        return self.bootstrap_validations == other.bootstrap_validations and self.metrics == other.metrics


class BoostValidation(Validation):
    def __init__(self, horizon: int, metrics: dict, predictions: dd, model: BaseEstimator):
        self.horizon = horizon
        super().__init__(metrics, predictions, model)

    def __eq__(self, other):
        return self.horizon == other.horizon and self.metrics == other.metrics and (
                self.predictions.compute().values == other.predictions.compute().values).all() and self.model == other.model


class ValidationSplit:
    def __init__(self, causal_validation: CausalValidation, split: str, boosted_validations: [BoostValidation],
                 truth: dd.DataFrame):
        self.causal_validation = causal_validation
        self.split = split
        self.truth = truth
        self.boosted_validations = boosted_validations

    def __eq__(self, other):
        return self.causal_validation == other.causal_validation and self.split == other.split and (
                self.truth.compute().values == other.truth.compute().values).all() and self.boosted_validations == other.boosted_validations


class PipelineValidation:
    def __init__(self, split_validations: [ValidationSplit]):
        self.split_validations = split_validations

    def __eq__(self, other):
        if not type(other) == PipelineValidation:
            return False
        return self.split_validations == other.split_validations


class Pipeline:
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
            validate_start=None,
            validate_end=None,
            validation_splits=None,
            link_function=None,
            confidence_intervals=None,
            random_seed=None,
            bootstrap_sample=None,
            scenarios=None,
            frequency=None,
            frequency_target_aggregation='sum',
            start=None,
            end=None,
            causal_model_type='GLM',
            causal_model_params=None,
            pipeline_root=None,
            boost_window=0,
            storage_options=None,
            boost_model_type='EWMA',
            boost_model_params=None,
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
        self.random_seed = random_seed
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
        self.validate_start = validate_start
        self.validate_end = validate_end
        self.validation_splits = validation_splits
        self.link_function = link_function
        self.target = target
        self.time_index = time_index
        self.scenarios = scenarios
        self.frequency = frequency
        self.frequency_target_aggregation = frequency_target_aggregation
        self.start = start
        self.end = end
        self.causal_model_type = causal_model_type
        self.causal_model_params = causal_model_params
        self.pipeline_root = pipeline_root
        self.boost_window = boost_window
        self.storage_options = storage_options
        self.boost_model_type = boost_model_type
        self.boost_model_params = boost_model_params
        self.bootstrap_models = []
        self.boost_models = {}
        self.is_fit = False

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

    @property
    def bootstrap(self):
        return self._bootstrap_sample > 0

    @_divina_component
    def preprocess(self, df: Union[str, dd.DataFrame], start=None, end=None,
                   dataset: Output = None):

        df[self.time_index] = dd.to_datetime(df[self.time_index])

        time_min, time_max = (
            df[self.time_index].min().compute(),
            df[self.time_index].max().compute(),
        )
        agg_map = {
                **{
                    c: "mean"
                    for c in df.columns
                    if df[c].dtype in [int, float] and c not in [self.time_index, self.target]
                },
                **{
                    c: "first"
                    for c in df.columns
                    if df[c].dtype not in [int, float] and c not in [self.time_index, self.target]
                },
                self.target: self.frequency_target_aggregation
            }

        def resample_agg(_df: pd.DataFrame):
            _columns = _df.columns
            _df = _df.set_index(self.time_index).resample(self.frequency).agg(agg_map)
            _df[self.target] = _df[self.target].fillna(0)
            _df = _df.fillna(method='ffill').reset_index()
            _df = _df[_columns]
            return _df


        if self.target_dimensions:
            df = df.groupby(self.target_dimensions).apply(resample_agg, meta=df.head(1))
        else:
            df = df.set_index(self.time_index).resample(self.frequency).agg(agg_map)
            df[self.target] = df[self.target].fillna(0)
            df = df.fillna(method='ffill').reset_index()
        if start:
            if pd.to_datetime(start) < time_min:
                raise Exception(
                    "Bad Start: {} < {} Check Dataset Time Range".format(
                        start, time_min
                    )
                )
            else:
                df = df[dd.to_datetime(df[self.time_index]) >= start]

        if end:
            if pd.to_datetime(end) > time_max:
                if not self.scenarios:
                    raise Exception(
                        "Bad End: {} | {} Check Dataset Time Range".format(
                            end, time_max
                        )
                    )
            else:
                df = df[dd.to_datetime(df[self.time_index]) <= end]

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
                    df["{}_({}, {}]".format(c, v, v_1)] = df[
                        "{}_({}, {}]".format(c, v, v_1)
                    ].where(((df[c] < v_1) & (df[c] >= v)), 0)

        if self.encode_features:
            for c in self.encode_features:
                if df[c].dtype == int:
                    df[c] = df[c].astype(float)
                else:
                    df[c] = df[c]
                df["{}_dummy".format(c)] = df[c]

            pipe = make_pipeline(
                Categorizer(columns=self.encode_features),
                DummyEncoder(columns=self.encode_features),
            )

            pipe.fit(df)

            df = pipe.transform(df)

            for c in self.encode_features:
                df[c] = df["{}_dummy".format(c)]
            df = df.drop(columns=["{}_dummy".format(c) for c in self.encode_features])

        if self.interaction_features:
            for t in self.interaction_features:
                if t in self.encode_features:
                    pipe = make_pipeline(
                        Categorizer(columns=[t]), DummyEncoder(columns=[t])
                    )
                    interactions = list(
                        pipe.fit(df[[t]]).steps[1][1].transformed_columns_
                    )
                else:
                    interactions = [t]
                for c in interactions:
                    for w in self.interaction_features[t]:
                        if w in self.encode_features:
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

        if self.encode_features:
            df = df.drop(columns=self.encode_features)

        if self.drop_features:
            df = df.drop(columns=self.drop_features)

        df[self.time_index] = dd.to_datetime(df[self.time_index])
        if self.target_dimensions:
            df["__target_dimension_index__"] = df[self.time_index].astype(str)
            for i, col in enumerate(self.target_dimensions):
                df["__target_dimension_index__"] += "__index__".format(i) + df[col].astype(str)
            df = df.drop(columns=[self.time_index] + self.target_dimensions)
            df = df.set_index("__target_dimension_index__")
        else:
            df = df.set_index(self.time_index)
        return df

    @_divina_component
    def split_dataset(self, split: str, df: Union[str, dd.DataFrame], train_df: Output = None,
                      test_df: Output = None):
        df = df.reset_index()
        if self.target_dimensions:
            expanded_df = df['__target_dimension_index__'].str.split('__index__', expand=True, n=len(df.head(
                1)['__target_dimension_index__'][
                0].split(
                '__index__')) - 1)
            expanded_df.columns = [self.time_index] + self.target_dimensions
            df = dd.merge(df, expanded_df)
            df = df.set_index('__target_dimension_index__').drop(columns=self.target_dimensions)
            df_train = df[(df[self.time_index] < split)].drop(columns=self.time_index)
            df_test = df[(df[self.time_index] >= split)].drop(columns=self.time_index)
        else:
            df_train = df[(df[self.time_index] < split)].set_index(self.time_index)
            df_test = df[(df[self.time_index] >= split)].set_index(self.time_index)
        return df_train, df_test

    @_divina_component
    def train(self, model_type: str,
              x: Union[str, dd.DataFrame],
              y: Union[str, dd.Series],
              random_state: int = 11,
              model_params: dict = None,
              bootstrap_percentage: float = None, horizon: int = 0, model: Output = None):

        if y.isna().sum().compute() > 0:
            raise ValueError('Null values in target not permitted.')

        if bootstrap_percentage:
            x = x.sample(
                replace=False, frac=bootstrap_percentage, random_state=random_state
            )
            y = y.sample(
                replace=False, frac=bootstrap_percentage, random_state=random_state
            )

        model = eval(model_type)()
        if model_params:
            for k in model_params:
                if not type(model_params[k]) in [list, np.array]:
                    model_params[k] = [model_params[k]]

            model = GridSearchCV(
                model,
                model_params,
                scoring=make_scorer(lambda y, y_hat: -abs(np.mean(y_hat - y)), greater_is_better=True)
            )
            model.fit(
                X=x.to_dask_array(lengths=True),
                y=y.shift(-horizon).dropna().to_dask_array(lengths=True),
                drop_constants=True
            )
            return model.best_estimator_

        model = eval(model_type)()
        model.fit(
            x.to_dask_array(lengths=True),
            y.shift(-horizon).dropna().to_dask_array(lengths=True),
            drop_constants=True
        )
        return model

    @_divina_component
    def forecast(self, model: Union[str, BaseEstimator], x: Union[str, dd.DataFrame],
                 predictions: Output = None):
        features = x.columns
        for f in features:
            if not is_numeric_dtype(x[f].dtype):
                try:
                    x[f] = x[f].astype(float)
                except ValueError:
                    raise ValueError(
                        "{} could not be converted to float. "
                        "Please convert to numeric or encode with "
                        "Experiment(encode_features=['{}'])".format(f, f)
                    )

        y_hat = dd.from_dask_array(model.predict(
            x.to_dask_array(lengths=True)
        )).to_frame()
        y_hat.index = x.index

        '''
        factor_df = dd.from_dask_array(
            df[features].to_dask_array(lengths=True)
            * da.from_array(model.linear_model.coef_)
        )
        factor_df.columns = ["factor_{}".format(c) for c in features]
        for c in factor_df:
            df[c] = factor_df[c]
        '''
        y_hat.columns = ['y_hat']
        return y_hat['y_hat']

    @_divina_component
    def validate(self, truth_dataset: Union[str, dd.DataFrame],
                 prediction_dataset: Union[str, dd.Series], metrics: Output = None):

        residuals = prediction_dataset - truth_dataset

        metrics = {"mse": residuals.pow(2).mean().compute()}

        return metrics

    @_divina_component
    def x_y_split(self, df: Union[str, dd.DataFrame], target=None, x: Output = None, y: Output = None):

        target = target or self.target
        x = df[[c for c in df.columns if not c == target]]
        y = df[target]

        return x, y

    @_divina_component
    def filter_and_join_predictions(self, x: Union[str, dd.DataFrame], y: Union[str, dd.DataFrame], start: str,
                                    end: str, filtered_df: Output = None):
        x['y_hat_boosted'] = y
        if self.target_dimensions:
            x = x.reset_index()
            expanded_x = x['__target_dimension_index__'].str.split('__index__', expand=True, n=len(x.head(
                1)['__target_dimension_index__'][
                0].split(
                '__index__')) - 1)
            expanded_x.columns = [self.time_index] + self.target_dimensions
            x = dd.merge(x, expanded_x).drop(columns=['__target_dimension_index__'])
        else:
            x = x.reset_index()
        return x[(x[self.time_index] >= start) & (x[self.time_index] <= end)]

    @_divina_component
    def aggregate_forecasts(self, forecasts: list, interval: Output = None,
                            point_estimates: Output = None):

        df_forecasts = dd.concat([dd.from_dask_array(s.to_dask_array(lengths=True)) for s in forecasts],
                                 axis=1)
        df_forecasts.columns = ['bootstrap_{}'.format(c) for c in range(len(forecasts))]
        df_interval = dd.from_dask_array(dd.from_dask_array(df_forecasts.to_dask_array(lengths=True).T).quantile(
            [i * 0.01 for i in self.confidence_intervals]).to_dask_array(lengths=True).T)

        df_interval.columns = [
            "{}_pred_c_{}".format(self.target, c)
            for c in self.confidence_intervals
        ]

        df_point_estimate = dd.from_dask_array(
            dd.from_dask_array(df_forecasts.to_dask_array(lengths=True).T).mean().to_dask_array(
                lengths=True).T)

        df_point_estimate.name = 'mean'

        df_interval = df_interval.repartition(
            divisions=df_forecasts.divisions
        )

        df_interval.index = forecasts[0].index

        df_point_estimate = df_point_estimate.repartition(
            divisions=df_forecasts.divisions
        )

        df_point_estimate.index = forecasts[0].index

        return df_interval, df_point_estimate

    @_divina_component
    def subtract_residuals(self, truth_series: Union[str, dd.Series], prediction_series: Union[str, dd.Series],
                           residuals: Output = None):

        residual_series = truth_series - prediction_series
        residual_series.name = prediction_series.name

        return residual_series

    @_divina_component
    def long_to_wide(self, series: Union[str, dd.Series],
                     horizon: int,
                     lag_df: Output = None):

        if len(series) < horizon + self.boost_window:
            raise ValueError('not enough observations to convert to wide format given window and horizon. shorten your window or horizon, or provide more truth data')

        if not series.name:
            series.name = 'series'
        series_name = series.name

        lags = range(horizon, horizon + self.boost_window)

        def _create_lags(series_name, lags, df):
            for lag in lags:
                df['lag_{}'.format(lag)] = df[series_name].shift(lag)
            return df

        if self.target_dimensions:
            target_dimensions_df = series.reset_index()['__target_dimension_index__'].str.split('__index__',
                                                                                                expand=True,
                                                                                                n=len(
                                                                                                    series.head(
                                                                                                        1).index[
                                                                                                        0].split(
                                                                                                        '__index__')) - 1)
            target_dimensions_df.index = series.index
            target_dimensions_df.columns = ['time'] + self.target_dimensions
            df = series.to_frame()
            for c in self.target_dimensions:
                df[c] = target_dimensions_df[c]
            lag_df = df.groupby(self.target_dimensions).apply(partial(_create_lags, series_name, lags)).drop(columns=self.target_dimensions).reset_index().set_index('__target_dimension_index__')

        else:
            lag_df = _create_lags(series_name, lags, series.to_frame())

        return lag_df

    @_divina_component
    def boost_forecast(self, forecast_series: Union[str, dd.Series],
                       adjustment_series: Union[str, dd.Series], boosted_forecast: Output = None):

        boosted_forecast = forecast_series + adjustment_series

        boosted_forecast.name = 'y_hat_boosted'

        return boosted_forecast

    @_divina_component
    def synthesize(self, df: Union[str, dd.DataFrame], scenario: dict, frequency: str, end: str, start: str = None,
                   synthetic_df: Output = None):

        supplied_columns = list(scenario.keys()) + [self.time_index, self.target]
        if self.target_dimensions:
            supplied_columns += self.target_dimensions

        if not all([c in supplied_columns for c in df.columns]):
            raise ValueError(
                'Not all columns passed in base dataframe present in scenario config. Please add a constant value or "last" for columns: {}'.format(
                    [c for c in df.columns if not c in supplied_columns]))

        npartitions = (df.memory_usage(deep=True).sum().compute() // 104857600) + 1

        df[self.time_index] = dd.to_datetime(df[self.time_index])

        time_min, time_max = (
            df[self.time_index].min().compute(),
            df[self.time_index].max().compute(),
        )
        if start and not pd.to_datetime(start) > time_max:
            raise ValueError('simulation start must be after the end of the truth data provided')
        if start:
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
        if not len(new_dates) > 0:
            raise ValueError('Check start and end - no dates to synthesize')
        combinations = list(new_dates)
        if self.target_dimensions:
            combinations = [
                list(x)
                for x in product(
                    combinations,
                    *[
                        df[s].unique().compute().values
                        for s in self.target_dimensions
                    ]
                )
            ]
            scenario_columns = [self.time_index] + self.target_dimensions
        else:
            combinations = [[x] for x in combinations]
            scenario_columns = [self.time_index]
        constant_columns = [
            c
            for c in scenario
        ]
        for c in constant_columns:
            combinations = [
                x[0] + [x[1]]
                for x in product(
                    combinations, [scenario[c]]
                )
            ]
        df_scenario = dd.from_pandas(
            pd.DataFrame(
                combinations, columns=scenario_columns + constant_columns
            ),
            npartitions=int(npartitions),
        )
        last_columns = [
            c for c in scenario if scenario[c] == "last"
        ]
        if len(last_columns) > 0:
            if self.target_dimensions:
                last = (
                    df.groupby(self.target_dimensions)[last_columns]
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
                    df_scenario.groupby(self.target_dimensions)
                    .apply(
                        partial(
                            join_func,
                            self.target_dimensions,
                            self.time_index,
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
                df.set_index(self.time_index),
                df_scenario.set_index(self.time_index),
            ],
            axis=0,
        ).reset_index()
        return df

    def fit(self, df, prefect=False):
        df = self.preprocess(df, prefect=prefect)

        validation_splits = []
        for s in self.validation_splits:
            train_df, test_df = self.split_dataset(df=df, split=s, prefect=prefect)
            bootstrap_predictions = []
            bootstrap_validations = []
            x_train, y_train = self.x_y_split(df=train_df, prefect=prefect)
            x_test, y_test = self.x_y_split(df=test_df, prefect=prefect)
            for n in self.bootstrap_seeds:
                bootstrap_model = self.train(model_type=self.causal_model_type,
                                             model_params=self.causal_model_params,
                                             x=x_train,
                                             y=y_train, bootstrap_percentage=0.8,
                                             random_state=n, prefect=prefect)

                bootstrap_prediction = self.forecast(x=x_test, model=bootstrap_model, prefect=prefect)
                bootstrap_validation = self.validate(truth_dataset=y_test,
                                                     prediction_dataset=bootstrap_prediction, prefect=prefect)
                self.bootstrap_models.append(bootstrap_model)
                bootstrap_predictions.append(bootstrap_prediction)
                bootstrap_validations.append(
                    Validation(metrics=bootstrap_validation, predictions=bootstrap_prediction,
                               model=bootstrap_model))
            confidence_intervals, point_estimates = self.aggregate_forecasts(bootstrap_predictions, prefect=prefect)
            causal_validation = self.validate(truth_dataset=y_test, prediction_dataset=point_estimates, prefect=prefect)
            boosted_validations = []
            residuals = self.subtract_residuals(prediction_series=point_estimates, truth_series=y_test, prefect=prefect)
            for h in self.time_horizons:
                wide_residuals = self.long_to_wide(series=residuals, horizon=h, prefect=prefect)
                x, y = self.x_y_split(df=wide_residuals, target='mean', prefect=prefect)
                boosted_model = self.train(model_type=self.boost_model_type,
                                           model_params=self.boost_model_params, x=x, y=y, prefect=prefect)
                residual_predictions = self.forecast(x=x, model=boosted_model, prefect=prefect)
                boosted_predictions = self.boost_forecast(forecast_series=point_estimates,
                                                          adjustment_series=residual_predictions, prefect=prefect)
                boosted_validation = self.validate(prediction_dataset=boosted_predictions,
                                                   truth_dataset=y_test, prefect=prefect)
                boosted_validations.append(
                    BoostValidation(metrics=boosted_validation, horizon=h, predictions=boosted_predictions,
                                    model=boosted_model))
                self.boost_models[h] = boosted_model

            validation_splits.append(
                ValidationSplit(
                    causal_validation=CausalValidation(metrics=causal_validation,
                                                       predictions=point_estimates,
                                                       bootstrap_validations=bootstrap_validations),
                    boosted_validations=boosted_validations,
                    split=s, truth=test_df))
        self.is_fit = True
        return PipelineValidation(split_validations=validation_splits)

    def simulate(self, truth, horizons: [int], scenarios: list, end: str, start: str, prefect=False):
        if not self.is_fit:
            raise ValueError('Pipeline must be fit before you can simulate.')
        scenario_preds = []
        for s in scenarios:
            ###TODO - make synthesize yield a single frequency
            synthetic_df = self.synthesize(df=truth, frequency='s', scenario=s, end=end, start=start, prefect=prefect)
            df = self.preprocess(df=synthetic_df, prefect=prefect)
            causal_x, y = self.x_y_split(df=df, prefect=prefect)
            bootstrap_predictions = []
            for m in self.bootstrap_models:
                ###TODO - implement map so that things run in parallel
                ###TODO then check the output of synthesize and write test
                bootstrap_prediction = self.forecast(x=causal_x, model=m, prefect=prefect)
                bootstrap_predictions.append(bootstrap_prediction)
            confidence_intervals, point_estimates = self.aggregate_forecasts(forecasts=bootstrap_predictions,
                                                                             prefect=prefect)
            boosted_preds = {}
            residuals = self.subtract_residuals(prediction_series=point_estimates, truth_series=y, prefect=prefect)
            for h, m in zip(horizons, [self.boost_models[h] for h in horizons]):
                wide_residuals = self.long_to_wide(series=residuals,
                                                   horizon=h, prefect=prefect)
                x, y = self.x_y_split(df=wide_residuals, target='mean', prefect=prefect)
                residual_predictions = self.forecast(x=x, model=m, prefect=prefect)
                boosted_predictions = self.boost_forecast(forecast_series=point_estimates,
                                                          adjustment_series=residual_predictions, prefect=prefect)
                filtered_predictions = self.filter_and_join_predictions(x=x, y=boosted_predictions, start=start,
                                                                        end=end)
                boosted_preds[h] = filtered_predictions
            scenario_preds.append(boosted_preds)
            ###TODO aggregate boosted predictions into single series
        return scenario_preds
