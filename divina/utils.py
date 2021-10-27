from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import dask.dataframe as dd
from dask_ml.linear_model import LinearRegression
from jsonschema import validate
from functools import wraps
import json
import pathlib


def compare_sk_models(model1, model2):
    if not isinstance(model1, Pipeline):
        steps1 = [("step1", model1)]
    else:
        steps1 = model1.steps
    if not isinstance(model2, Pipeline):
        steps2 = [("step1", model2)]
    else:
        steps2 = model2.steps
    for s, o in zip(steps1, steps2):
        for i, j in zip(s, o):
            assert type(i) == type(j)
            if isinstance(i, BaseEstimator):
                assert set(i.get_params()) == set(j.get_params())
            if isinstance(i, LinearRegression):
                assert i.coef_ == j.coef_
                assert i.intercept_ == j.intercept_
            return None


def cull_empty_partitions(df):
    ll = list(df.map_partitions(len).compute())
    df_delayed = df.to_delayed()
    df_delayed_new = list()
    pempty = None
    for ix, n in enumerate(ll):
        if 0 == n:
            pempty = df.get_partition(ix)
        else:
            df_delayed_new.append(df_delayed[ix])
    if pempty is not None:
        df = dd.from_delayed(df_delayed_new, meta=pempty)
    return df


def validate_forecast_definition(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with open(pathlib.Path(pathlib.Path(__file__).parent, 'config/fd_schema.json'), 'r') as f:
            validate(instance={'forecast_definition': kwargs['forecast_definition']}, schema=json.load(f))
        return func(*args, **kwargs)

    return wrapper
