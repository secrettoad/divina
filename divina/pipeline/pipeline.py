import time
from datetime import datetime
from functools import partial
from itertools import zip_longest
from typing import Union

import dask.dataframe as dd
import pandas as pd
from dask_ml.preprocessing import Categorizer, DummyEncoder
from pandas.api.types import is_numeric_dtype
from pandas.testing import assert_frame_equal, assert_series_equal
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.pipeline import make_pipeline

from divina.divina.pipeline.utils import (
    Output,
    _divina_component,
    get_dask_client,
    cull_empty_partitions,
)

from .model import *  # noqa: F403, F401

supports_factors = ["GLM"]


class CausalPrediction:
    def __init__(
        self,
        factors: dd.DataFrame,
        predictions: dd.DataFrame,
        confidence_intervals: dd.DataFrame,
    ):
        self.predictions = predictions
        self.confidence_intervals = confidence_intervals
        self.factors = factors

    def __eq__(self, other):
        return (
            self.factors == other.factors
            and (
                self.predictions.compute().values == other.predictions.compute().values
            ).all()
            and self.confidence_intervals == other.confidence_intervals
        )


class BoostPrediction:
    def __init__(
        self,
        horizon: int,
        causal_predictions: CausalPrediction,
        residual_predictions: dd.DataFrame,
        predictions: dd.DataFrame,
        confidence_intervals: dd.DataFrame,
        model: BaseEstimator,
        lag_features: dd.DataFrame,
    ):
        self.predictions = predictions
        self.residual_predictions = residual_predictions
        self.confidence_intervals = confidence_intervals
        self.model = model
        self.lag_features = lag_features
        self.causal_predictions = causal_predictions
        self.horizon = horizon

    def __eq__(self, other):
        return (
            (
                self.confidence_intervals.compute().values
                == other.confidence_intervals.compute().values
            ).all()
            and self.horizon == other.horizon
            and self.model == other.model
            and (
                self.predictions.compute().values == other.predictions.compute().values
            ).all()
            and (
                self.lag_features.compute().values
                == other.lag_features.compute().values
            ).all()
            and (
                self.residual_predictions.compute().values
                == other.residual_predictions.compute().values
            ).all()
            and self.causal_predictions == other.causal_predictions
        )


class PipelinePredictResult:
    def __init__(
        self,
        causal_predictions: CausalPrediction,
        truth: dd.DataFrame,
        boost_predictions: [BoostPrediction] = None,
    ):
        self.boost_predictions = boost_predictions
        self.causal_predictions = causal_predictions
        self.truth = truth

    def __eq__(self, other):
        if not type(other) == PipelinePredictResult:
            return False
        return (
            self.boost_predictions == other.boost_predictions
            and self.causal_predictions == other.causal_predictions
        )

    def __getitem__(self, key):
        if key == 0:
            return self.causal_predictions
        else:
            return [p for p in self.boost_predictions if p.horizon == key][0]

    def __len__(self):
        return len(self.boost_predictions) + 1

    def __iter__(self):
        for p in [self.causal_predictions] + self.boost_predictions:
            yield p


class Validation:
    def __init__(
        self,
        metrics: dict,
        predictions: dd.DataFrame,
        factors: dd.DataFrame = None,
        model: BaseEstimator = None,
    ):
        self.metrics = metrics
        self.predictions = predictions
        self.factors = factors
        self.model = model

    def __eq__(self, other):
        return (
            self.metrics == other.metrics
            and (
                self.predictions.compute().values == other.predictions.compute().values
            ).all()
            and self.model == other.model
        )


class CausalValidation(Validation):
    def __init__(
        self,
        bootstrap_validations: [Validation],
        metrics: dict,
        predictions: dd.DataFrame,
        factors: dd.DataFrame = None,
    ):
        self.bootstrap_validations = bootstrap_validations
        super().__init__(metrics=metrics, predictions=predictions, factors=factors)

    def __eq__(self, other):
        return (
            self.bootstrap_validations == other.bootstrap_validations
            and super().__eq__(other)
        )


class BoostValidation(Validation):
    def __init__(
        self,
        horizon: int,
        metrics: dict,
        predictions: dd.DataFrame,
        model: BaseEstimator,
        residual_predictions: dd.DataFrame = None,
        factors: dd.DataFrame = None,
    ):
        self.horizon = horizon
        self.residual_predictions = residual_predictions
        super().__init__(
            metrics=metrics,
            predictions=predictions,
            model=model,
            factors=factors,
        )

    def __eq__(self, other):
        return self.horizon == other.horizon and super().__eq__(other)


class ValidationSplit:
    def __init__(
        self,
        causal_validation: CausalValidation,
        truth: dd.DataFrame,
        split: str = None,
        boosted_validations: [BoostValidation] = None,
    ):
        self.causal_validation = causal_validation
        self.split = split
        self.truth = truth
        self.boosted_validations = boosted_validations

    def __eq__(self, other):
        return (
            self.causal_validation == other.causal_validation
            and self.split == other.split
            and (self.truth.compute().values == other.truth.compute().values).all()
            and self.boosted_validations == other.boosted_validations
        )


class PipelineFitResult:
    def __init__(self, split_validations: [ValidationSplit], *args):
        self.split_validations = split_validations
        self.tpl = args

    def __eq__(self, other):
        if not type(other) == PipelineFitResult:
            return False
        return self.split_validations == other.split_validations

    def __getitem__(self, key):
        return self.split_validations[key]

    def __len__(self):
        return len(self.split_validations)

    def __hash__(self):
        return hash(self.tpl)

    def __repr__(self):
        return repr(self.tpl)


def assert_pipeline_fit_result_equal(pr1: PipelineFitResult, pr2: PipelineFitResult):
    for s1, s2 in zip_longest(pr1, pr2):
        assert s1.split == s2.split
        for bs1, bs2 in zip_longest(
            s1.causal_validation.bootstrap_validations,
            s2.causal_validation.bootstrap_validations,
        ):
            assert bs1.model == bs2.model
            assert_series_equal(bs1.predictions.compute(), bs2.predictions.compute())
            print(bs1.metrics)
            print(bs2.metrics)
            assert bs1.metrics == bs2.metrics
        assert_frame_equal(s1.truth.compute(), s2.truth.compute())
        assert_series_equal(
            s1.causal_validation.predictions.compute(),
            s2.causal_validation.predictions.compute(),
        )
        assert s1.causal_validation.metrics == s2.causal_validation.metrics
        for b1, b2 in zip_longest(s1.boosted_validations, s2.boosted_validations):
            assert b1.horizon == b2.horizon
            assert b1.model == b2.model
            assert_series_equal(b1.predictions.compute(), b2.predictions.compute())
            assert b1.metrics == b2.metrics


def assert_pipeline_predict_result_equal(
    pr1: PipelinePredictResult, pr2: PipelinePredictResult
):
    assert_frame_equal(pr1.truth.compute(), pr2.truth.compute())
    for s1, s2 in zip_longest(pr1, pr2):
        if type(s1) == CausalPrediction:
            assert type(s2) == CausalPrediction
            assert_series_equal(s1.predictions.compute(), s2.predictions.compute())
            assert_frame_equal(s1.factors.compute(), s2.factors.compute())
            assert_frame_equal(
                s1.confidence_intervals.compute(),
                s2.confidence_intervals.compute(),
            )
        elif type(s1) == BoostPrediction:
            assert type(s2) == BoostPrediction
            assert s1.model == s2.model
            assert s1.horizon == s1.horizon
            assert_series_equal(s1.predictions.compute(), s2.predictions.compute())
            assert_frame_equal(
                s1.confidence_intervals.compute(),
                s2.confidence_intervals.compute(),
            )
            assert_frame_equal(s1.lag_features.compute(), s2.lag_features.compute())
            assert_series_equal(
                s1.residual_predictions.compute(),
                s2.residual_predictions.compute(),
            )


class Pipeline:
    def __init__(
        self,
        target,
        time_index,
        frequency,
        target_dimensions=None,
        include_features=None,
        drop_features=None,
        time_features=False,
        encode_features=None,
        bin_features=None,
        interaction_features=None,
        time_horizons=None,
        validation_splits=None,
        link_function=None,
        confidence_intervals=None,
        random_seed=None,
        bootstrap_sample=None,
        scenarios=None,
        frequency_target_aggregation="sum",
        causal_model_type="GLM",
        causal_model_params=None,
        pipeline_root=None,
        boost_window=0,
        storage_options=None,
        boost_model_type="EWMA",
        boost_model_params=None,
    ):
        if not time_horizons:
            time_horizons = []
        if not causal_model_params:
            causal_model_params = [None]
        horizon_ranges = [x for x in time_horizons if type(x) == tuple]
        if len(horizon_ranges) > 0:
            self.time_horizons = [x for x in time_horizons if type(x) == int]
            for x in horizon_ranges:
                self.time_horizons = set(self.time_horizons + list(range(x[0], x[1])))
        else:
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
        self.time_features = time_features
        self.include_features = include_features
        self.drop_features = drop_features
        self.encode_features = encode_features
        self.bin_features = bin_features
        self.interaction_features = interaction_features
        self.validation_splits = validation_splits
        self.link_function = link_function
        self.target = target
        self.time_index = time_index
        self.scenarios = scenarios
        self.frequency = frequency
        self.frequency_target_aggregation = frequency_target_aggregation
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
        self.fit_features = None
        self._target_dimension_dtypes = None

    @property
    def bootstrap_sample(self):
        return self._bootstrap_sample

    @bootstrap_sample.setter
    def bootstrap_sample(self, value):
        self._bootstrap_sample = value
        if hasattr(self, "random_seed") and self.random_seed:
            self.bootstrap_seeds = [
                x for x in range(self.random_seed, self.random_seed + value)
            ]
        else:
            self.bootstrap_seeds = [x for x in np.random.randint(0, 10000, size=value)]
        return

    @property
    def bootstrap(self):
        return self._bootstrap_sample > 0

    def extract_dask_multiindex(self, df):
        if type(df) == dd.Series:
            df = df.to_frame()
        expanded_df = (
            df.reset_index()
            .set_index("__target_dimension_index__", drop=False)[
                "__target_dimension_index__"
            ]
            .str.split(
                "__index__",
                expand=True,
                n=len(
                    df.head(1)
                    .reset_index()["__target_dimension_index__"][0]
                    .split("__index__")
                )
                - 1,
            )
        )
        expanded_df.columns = [self.time_index] + self.target_dimensions
        for _c in expanded_df:
            df[_c] = expanded_df[_c]
        df[self.time_index] = dd.to_datetime(df[self.time_index])
        for _c, d in zip(self.target_dimensions, self._target_dimension_dtypes):
            df[_c] = df[_c].astype(d)
        return df

    def set_dask_multiindex(self, df):
        df["__target_dimension_index__"] = df[self.time_index].astype(str)
        for i, col in enumerate(self.target_dimensions):
            df["__target_dimension_index__"] += "__index__" + df[col].astype(str)
        df = df.drop(columns=[self.time_index] + self.target_dimensions)
        df = df.set_index("__target_dimension_index__")
        return df

    @_divina_component
    def preprocess(
        self,
        df: Union[str, dd.DataFrame],
        start=None,
        end=None,
        dataset: Output = None,
    ):

        df[self.time_index] = dd.to_datetime(df[self.time_index])

        time_min, time_max = (
            df[self.time_index].min().compute(),
            df[self.time_index].max().compute(),
        )

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

        agg_map = {
            **{
                c: "mean"
                for c in df.columns
                if df[c].dtype in [int, float]
                and c not in [self.time_index, self.target]
            },
            **{
                c: "last"
                for c in df.columns
                if df[c].dtype not in [int, float]
                and c not in [self.time_index, self.target]
            },
        }
        if self.target in df.columns:
            agg_map.update({self.target: self.frequency_target_aggregation})

        # TODO - ADD WARNING ON DUPLICATES BEFORE AGG
        if self.target_dimensions:
            df = (
                df.groupby([self.time_index] + self.target_dimensions)
                .agg(agg_map)
                .drop(columns=self.target_dimensions)
                .reset_index()
            )
        else:
            df = df.groupby([self.time_index]).aggregate(agg_map).reset_index()

        if self.time_features:
            df[self.time_index] = dd.to_datetime(df[self.time_index])

            df["Month"] = df[self.time_index].dt.month
            df["Day"] = df[self.time_index].dt.dayofyear
            df["Year"] = df[self.time_index].dt.year
            df["Weekday"] = df[self.time_index].dt.weekday
            df["T"] = (
                df[self.time_index]
                - pd.to_datetime(datetime.fromtimestamp(time.mktime(time.gmtime(0))))
            ) / pd.to_timedelta("1{}".format(self.frequency))

            cal = calendar()
            holidays = cal.holidays(start=time_min, end=time_max, return_name=True)

            df["Holiday"] = (
                df[self.time_index]
                .apply(lambda x: holidays.get(x))
                .astype(bool)
                .astype(int)
            )
            df["HolidayType"] = df[self.time_index].apply(lambda x: holidays.get(x))

            df["LastDayOfMonth"] = (
                df[self.time_index].dt.daysinmonth == df[self.time_index].dt.day
            ).astype(int)

            df["DayOfMonth"] = df[self.time_index].dt.day

            df["WeekOfYear"] = df[self.time_index].dt.week

            if not self.encode_features:
                self.encode_features = ["HolidayType"]
            elif "HolidayType" not in self.encode_features:
                self.encode_features += ["HolidayType"]
            if not self.drop_features:
                self.drop_features = ["HolidayType"]
            elif "HolidayType" not in self.drop_features:
                self.drop_features += ["HolidayType"]

        for c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype(float)

        if self.include_features:
            df = (
                df[[self.target, self.time_index] + self.include_features]
                if self.target in df.columns
                else df[[self.time_index] + self.include_features]
            )

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
            _columns = []
            for t in self.interaction_features:
                for m in self.interaction_features[t]:
                    if "{}-x-{}".format(m, t) in df.columns:
                        column_name = "{}-x-{}".format(m, t)
                    else:
                        column_name = "{}-x-{}".format(t, m)
                    df[column_name] = (
                        t + "-" + df[t].astype(str) + "-" + m + "-" + df[m].astype(str)
                    )
                    _columns.append(column_name)
            pipe = make_pipeline(
                Categorizer(columns=_columns),
                DummyEncoder(columns=_columns, drop_first=True),
            )
            pipe.fit(df)
            df = pipe.transform(df)

        if self.drop_features:
            df = df.drop(columns=self.drop_features)

        df[self.time_index] = dd.to_datetime(df[self.time_index])
        if self.target_dimensions:
            self._target_dimension_dtypes = list(df[self.target_dimensions].dtypes)
            # TODO - issue stems from setting multiindex
            df = self.set_dask_multiindex(df)
        else:
            df = df.set_index(self.time_index)
        if self.fit_features is None:
            self.fit_features = list(df.columns)
        else:
            for c in self.fit_features:
                if c not in df.columns:
                    df[c] = 0
        return df

    @_divina_component
    def split_dataset(
        self,
        split: str,
        df: Union[str, dd.DataFrame],
        train_df: Output = None,
        test_df: Output = None,
    ):
        if self.target_dimensions:
            df = self.extract_dask_multiindex(df)
            df_train = df[(dd.to_datetime(df[self.time_index]) < split)].drop(
                columns=[self.time_index] + self.target_dimensions
            )
            df_test = df[(dd.to_datetime(df[self.time_index]) >= split)].drop(
                columns=[self.time_index] + self.target_dimensions
            )
        else:
            df_train = df[(dd.to_datetime(df[self.time_index]) < split)].set_index(
                self.time_index
            )
            df_test = df[(dd.to_datetime(df[self.time_index]) >= split)].set_index(
                self.time_index
            )
        df_test = (
            cull_empty_partitions(df_test).reset_index().set_index(df_test.index.name)
        )
        df_train = (
            cull_empty_partitions(df_train).reset_index().set_index(df_train.index.name)
        )
        return df_train, df_test

    @_divina_component
    def train(
        self,
        model_type: str,
        x: Union[str, dd.DataFrame],
        y: Union[str, dd.Series],
        random_state: int = 11,
        model_params: dict = None,
        bootstrap_percentage: float = None,
        horizon: int = 0,
        model: Output = None,
    ):

        if y.isna().sum().compute() > 0:
            raise ValueError("Null values in target not permitted.")
        if bootstrap_percentage:
            x = x.sample(
                replace=False,
                frac=bootstrap_percentage,
                random_state=random_state,
            )
            y = y.sample(
                replace=False,
                frac=bootstrap_percentage,
                random_state=random_state,
            )
            x = cull_empty_partitions(x)
            y = cull_empty_partitions(y)

        if model_params:
            model = eval(model_type)(**model_params)
        else:
            model = eval(model_type)()
        model.fit(x, y.shift(-horizon).dropna(), drop_constants=True)

        return model

    @_divina_component
    def forecast(
        self,
        model: Union[str, BaseEstimator],
        x: Union[str, dd.DataFrame],
        predictions: Output = None,
    ):
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

        y_hat = dd.from_dask_array(
            model.predict(x.to_dask_array(lengths=True))
        ).to_frame()
        y_hat.index = x.index
        y_hat.columns = ["y_hat"]

        return y_hat["y_hat"]

    @_divina_component
    def validate(
        self,
        truth_dataset: Union[str, dd.DataFrame],
        prediction_dataset: Union[str, dd.Series],
        metrics: Output = None,
    ):

        residuals = prediction_dataset - truth_dataset

        metrics = {"mse": residuals.pow(2).mean().compute()}

        return metrics

    @_divina_component
    def x_y_split(
        self,
        df: Union[str, dd.DataFrame],
        target=None,
        x: Output = None,
        y: Output = None,
    ):

        target = target or self.target
        x = df[[c for c in df.columns if not c == target]]
        y = df[target]

        return x, y

    @_divina_component
    def aggregate_forecasts(
        self,
        forecasts: list,
        interval: Output = None,
        point_estimates: Output = None,
    ):

        df_forecasts = dd.concat(
            [dd.from_dask_array(s.to_dask_array(lengths=True)) for s in forecasts],
            axis=1,
        )
        df_forecasts.columns = ["bootstrap_{}".format(c) for c in range(len(forecasts))]
        if self.confidence_intervals:
            df_interval = dd.from_dask_array(
                dd.from_dask_array(df_forecasts.to_dask_array(lengths=True).T)
                .quantile([i * 0.01 for i in self.confidence_intervals])
                .to_dask_array(lengths=True)
                .T
            )

            df_interval.columns = [
                "{}_pred_c_{}".format(self.target, c) for c in self.confidence_intervals
            ]
            df_interval = df_interval.repartition(divisions=df_forecasts.divisions)

            df_interval.index = forecasts[0].index
        else:
            df_interval = None

        df_point_estimate = dd.from_dask_array(
            dd.from_dask_array(df_forecasts.to_dask_array(lengths=True).T)
            .mean()
            .to_dask_array(lengths=True)
            .T
        )

        df_point_estimate.name = "y_hat"

        df_point_estimate = df_point_estimate.repartition(
            divisions=df_forecasts.divisions
        )

        df_point_estimate.index = forecasts[0].index

        return df_interval, df_point_estimate

    @_divina_component
    def aggregate_factors(self, factors: list, aggregated_factors: Output = None):
        aggregated_factors = factors[0]
        for df in factors[1:]:
            aggregated_factors += df
        aggregated_factors = aggregated_factors / len(factors)

        return aggregated_factors

    @_divina_component
    def multiply_factors(
        self,
        x: Union[str, dd.DataFrame],
        model: Union[str, BaseEstimator],
        factors: Output = None,
    ):
        if not hasattr(model, "coef_"):
            raise ValueError(
                "Model provided does not have coef_ "
                "attribute. Cannot calculate factors."
            )

        features = (
            x.columns
            if not model.fit_indices
            else [x.columns[i] for i in model.fit_indices]
        )

        factors = dd.from_dask_array(
            x[features].to_dask_array(lengths=True)
            * da.from_array(model.linear_model.coef_)
        )
        factors.columns = ["factor_{}".format(c) for c in features]

        factors.index = x.index

        return factors

    @_divina_component
    def subtract_residuals(
        self,
        truth_series: Union[str, dd.Series],
        prediction_series: Union[str, dd.Series],
        residuals: Output = None,
    ):

        residual_series = truth_series - prediction_series
        residual_series.name = prediction_series.name

        return residual_series

    @_divina_component
    def long_to_wide(
        self,
        series: Union[str, dd.Series],
        horizon: int,
        lag_df: Output = None,
    ):

        if not series.name:
            series.name = "series"
        series_name = series.name

        lags = range(horizon, horizon + self.boost_window)

        def resample_shift(lag, _series_name, _df: pd.DataFrame):
            _columns = _df.columns
            if not _df.index.name:
                _df.index.name = "index"
            _index_name = _df.index.name
            _index = _df.index
            _df = _df.reset_index().set_index(self.time_index)
            _df = (
                _df.resample(self.frequency)
                .asfreq()
                .reset_index()
                .set_index(_index_name)
            )
            _df[_series_name] = _df[_series_name].shift(lag)
            _df = _df.loc[_index]
            _df = _df[_columns]
            return _df

        if self.target_dimensions:
            target_dimensions_df = series.reset_index()[
                "__target_dimension_index__"
            ].str.split(
                "__index__",
                expand=True,
                n=len(series.head(1).index[0].split("__index__")) - 1,
            )
            target_dimensions_df.index = series.index
            target_dimensions_df.columns = [self.time_index] + self.target_dimensions
            df = series.to_frame()
            for c in target_dimensions_df.columns:
                df[c] = target_dimensions_df[c]
            df[self.time_index] = dd.to_datetime(df[self.time_index])
            for lag in lags:
                df["lag_{}".format(lag)] = (
                    df.groupby(self.target_dimensions)
                    .apply(
                        partial(resample_shift, lag, series_name),
                        meta=df.head(1),
                    )
                    .reset_index()
                    .set_index("__target_dimension_index__")[series_name]
                )
            df = (
                df.drop(columns=[self.time_index] + self.target_dimensions)
                .reset_index()
                .set_index("__target_dimension_index__")
            )

        else:
            df = series.to_frame()
            for lag in lags:
                df["lag_{}".format(lag)] = resample_shift(lag, df, series_name)[
                    series_name
                ]

        return df

    @_divina_component
    def boost_forecast(
        self,
        forecast_series: Union[str, dd.Series],
        adjustment_series: Union[str, dd.Series],
        boosted_forecast: Output = None,
    ):

        boosted_forecast = forecast_series + adjustment_series.fillna(0)

        boosted_forecast.name = "y_hat_boosted"

        return boosted_forecast

    @_divina_component
    def add_boost_to_intervals(
        self,
        adjustment_series: Union[str, dd.Series],
        intervals: Union[str, dd.DataFrame],
        boosted_intervals: Output = None,
    ):

        for c in intervals.columns:
            intervals[c] = intervals[c] + adjustment_series.fillna(0)

        boosted_intervals = intervals

        return boosted_intervals

    @get_dask_client
    def fit(self, df, start=None, end=None, prefect=False, dask_configuration=None):
        def _causal_fit(
            x_train,
            y_train,
            x_test,
            y_test,
            random_state,
            prefect,
            model_params,
            bootstrap_percentage=None,
        ):
            bootstrap_model = self.train(
                model_type=self.causal_model_type,
                model_params=model_params,
                x=x_train,
                y=y_train,
                bootstrap_percentage=bootstrap_percentage,
                random_state=random_state,
                prefect=prefect,
            )

            bootstrap_prediction = self.forecast(
                x=x_test, model=bootstrap_model, prefect=prefect
            )
            if self.causal_model_type in supports_factors:
                bootstrap_factors = self.multiply_factors(
                    x=x_test, model=bootstrap_model
                )
            else:
                bootstrap_factors = None
            bootstrap_validation = self.validate(
                truth_dataset=y_test,
                prediction_dataset=bootstrap_prediction,
                prefect=prefect,
            )
            return Validation(
                metrics=bootstrap_validation,
                predictions=bootstrap_prediction,
                factors=bootstrap_factors,
                model=bootstrap_model,
            )

        def _fit(train_df, test_df, prefect):
            bootstrap_validations = []
            x_train, y_train = self.x_y_split(df=train_df, prefect=prefect)
            x_test, y_test = self.x_y_split(df=test_df, prefect=prefect)
            if self.bootstrap_seeds:
                for n in self.bootstrap_seeds:
                    cv_validations = []
                    for c in self.causal_model_params:
                        cv_validations.append(
                            _causal_fit(
                                x_train=x_train,
                                y_train=y_train,
                                x_test=x_test,
                                y_test=y_test,
                                random_state=n,
                                bootstrap_percentage=0.8,
                                model_params=c,
                                prefect=prefect,
                            )
                        )
                    best_cv_validation = max(
                        cv_validations, key=lambda _cv: _cv.metrics["mse"]
                    )
                    bootstrap_validations.append(best_cv_validation)
                    self.bootstrap_models.append(best_cv_validation.model)
                (confidence_intervals, point_estimates,) = self.aggregate_forecasts(
                    [v.predictions for v in bootstrap_validations],
                    prefect=prefect,
                )
                if self.causal_model_type in supports_factors:
                    factors = self.aggregate_factors(
                        [v.factors for v in bootstrap_validations]
                    )
                else:
                    factors = None
            else:
                cv_validations = []
                for c in self.causal_model_params:
                    cv_validations.append(
                        _causal_fit(
                            x_train=x_train,
                            y_train=y_train,
                            x_test=x_test,
                            y_test=y_test,
                            random_state=11,
                            bootstrap_percentage=0.8,
                            model_params=c,
                            prefect=prefect,
                        )
                    )
                # TODO - get this to properly paralellize on
                #  prefect - map instead of for
                # TODO - add cv validations to result object
                best_cv_validation = min(
                    cv_validations, key=lambda _cv: _cv.metrics["mse"]
                )
                self.bootstrap_models.append(best_cv_validation.model)
                point_estimates = best_cv_validation.predictions
                if self.causal_model_type in supports_factors:
                    factors = best_cv_validation.factors
                else:
                    factors = None
            causal_validation = self.validate(
                truth_dataset=y_test,
                prediction_dataset=point_estimates,
                prefect=prefect,
            )
            validation_split = ValidationSplit(
                causal_validation=CausalValidation(
                    metrics=causal_validation,
                    predictions=point_estimates,
                    factors=factors,
                    bootstrap_validations=bootstrap_validations,
                ),
                truth=test_df,
            )
            if len(self.time_horizons) > 0:
                boosted_validations = []
                residuals = self.subtract_residuals(
                    prediction_series=point_estimates,
                    truth_series=y_test,
                    prefect=prefect,
                )
                for h in self.time_horizons:
                    wide_residuals = self.long_to_wide(
                        series=residuals, horizon=h, prefect=prefect
                    )
                    x, y = self.x_y_split(
                        df=wide_residuals, target="y_hat", prefect=prefect
                    )
                    boosted_model = self.train(
                        model_type=self.boost_model_type,
                        model_params=self.boost_model_params,
                        x=x,
                        y=y,
                        prefect=prefect,
                    )
                    residual_predictions = self.forecast(
                        x=x, model=boosted_model, prefect=prefect
                    )
                    boosted_predictions = self.boost_forecast(
                        forecast_series=point_estimates,
                        adjustment_series=residual_predictions,
                        prefect=prefect,
                    )
                    boosted_validation = self.validate(
                        prediction_dataset=boosted_predictions,
                        truth_dataset=y_test,
                        prefect=prefect,
                    )
                    boosted_validations.append(
                        BoostValidation(
                            metrics=boosted_validation,
                            horizon=h,
                            predictions=boosted_predictions,
                            residual_predictions=residual_predictions,
                            model=boosted_model,
                        )
                    )
                    self.boost_models[h] = boosted_model
                validation_split.boosted_validations = boosted_validations
            return validation_split

        df = self.preprocess(df, prefect=prefect, start=start, end=end)
        self.fit_features = df.columns
        validation_splits = []
        if self.validation_splits:
            for s in self.validation_splits:
                train_df, test_df = self.split_dataset(df=df, split=s, prefect=prefect)
                validation_split = _fit(train_df, test_df, prefect)
                validation_split.split = s
                validation_splits.append(validation_split)
        else:
            validation_splits.append(_fit(df, df, prefect))
        self.is_fit = True
        return PipelineFitResult(split_validations=validation_splits)

    @get_dask_client
    def predict(
        self,
        x: dd.DataFrame,
        horizons: [int] = None,
        boost_y: str = None,
        prefect=False,
        dask_configuration=None,
    ):
        if not self.is_fit:
            raise ValueError("Pipeline must be fit before you can predict.")
        if horizons and len(set(horizons) - set(self.time_horizons)) > 0:
            raise ValueError(
                "Pipeline not train on horizons: "
                "{}. Train with pipeline.fit()".format(
                    set(horizons) - set(self.time_horizons)
                )
            )
        x = self.preprocess(df=x, prefect=prefect)
        if boost_y:
            x, y = self.x_y_split(df=x, target=boost_y, prefect=prefect)
        bootstrap_predictions = []
        bootstrap_factors = []
        for m in self.bootstrap_models:
            # TODO - implement map so that things run in parallel
            bootstrap_prediction = self.forecast(x=x, model=m, prefect=prefect)
            bootstrap_predictions.append(bootstrap_prediction)

            if self.causal_model_type in supports_factors:
                bootstrap_factors.append(self.multiply_factors(x=x, model=m))

        if len(bootstrap_factors) > 1:
            factors = self.aggregate_factors(bootstrap_factors)
        elif len(bootstrap_factors) == 1:
            factors = bootstrap_factors[0]
        else:
            factors = None
        intervals, point_estimates = self.aggregate_forecasts(
            forecasts=bootstrap_predictions, prefect=prefect
        )
        causal_prediction = CausalPrediction(
            factors=factors,
            predictions=point_estimates,
            confidence_intervals=intervals,
        )
        if boost_y:
            boosted_prediction_results = []
            residuals = self.subtract_residuals(
                prediction_series=point_estimates,
                truth_series=y,
                prefect=prefect,
            )
            for h, m in zip(horizons, [self.boost_models[h] for h in horizons]):
                wide_residuals = self.long_to_wide(
                    series=residuals, horizon=h, prefect=prefect
                )
                x_wide, y = self.x_y_split(
                    df=wide_residuals, target="y_hat", prefect=prefect
                )
                residual_predictions = self.forecast(x=x_wide, model=m, prefect=prefect)
                boosted_predictions = self.boost_forecast(
                    forecast_series=point_estimates,
                    adjustment_series=residual_predictions,
                    prefect=prefect,
                )
                if self.confidence_intervals:
                    boost_confidence_intervals = self.add_boost_to_intervals(
                        intervals=intervals,
                        adjustment_series=residual_predictions,
                        prefect=prefect,
                    )
                else:
                    boost_confidence_intervals = None
                boosted_prediction_results.append(
                    BoostPrediction(
                        horizon=h,
                        causal_predictions=causal_prediction,
                        residual_predictions=residual_predictions,
                        predictions=boosted_predictions,
                        confidence_intervals=boost_confidence_intervals,
                        model=m,
                        lag_features=x_wide,
                    )
                )

                # TODO aggregate boosted predictions into single series
            return PipelinePredictResult(
                causal_predictions=causal_prediction,
                truth=x,
                boost_predictions=boosted_prediction_results,
            )
        else:
            return PipelinePredictResult(causal_predictions=causal_prediction, truth=x)
