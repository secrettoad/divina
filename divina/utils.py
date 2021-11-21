from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import dask.dataframe as dd
from dask_ml.linear_model import LinearRegression
from jsonschema import validate
from functools import wraps
import pathlib
import numpy as np
import json
import os


def get_parameters(s3_fs, model_path):
    if model_path[:5] == "s3://":
        if not s3_fs.exists(model_path):
            s3_fs.mkdir(
                model_path,
                create_parents=True,
                region_name=os.environ["AWS_DEFAULT_REGION"],
                acl="private",
            )
        write_open = s3_fs.open

    else:
        write_open = open
    with write_open(
            '{}_params'.format(model_path),
            "rb"
    ) as f:
        params = json.load(f)
        return params


def set_parameters(s3_fs, model_path, params):
    if model_path[:5] == "s3://":
        if not s3_fs.exists(model_path):
            s3_fs.mkdir(
                model_path,
                create_parents=True,
                region_name=os.environ["AWS_DEFAULT_REGION"],
                acl="private",
            )
        write_open = s3_fs.open

    else:
        write_open = open
    with write_open(
            '{}_params'.format(model_path),
            "rb"
    ) as f:
        parameters = json.load(f)['features']
    if not params == parameters:
        raise Exception('Parameters {} not found in trained model. Cannot set new values for these parameters'.format(
            ', '.join(list(set(params) - set(parameters)))))
    else:
        for p in params:
            if not p in parameters:
                parameters.append(p)
        with write_open(
                '{}_params'.format(model_path),
                "w"
        ) as f:
            json.dump({'features': parameters}, f)


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
        if isinstance(s[1], BaseEstimator):
            assert set(s[1].get_params()) == set(o[1].get_params())
        if isinstance(s[1], LinearRegression):
            assert np.allclose(s[1].coef_, o[1].coef_)
            assert np.allclose([s[1].intercept_], [o[1].intercept_])
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


def validate_experiment_definition(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with open(pathlib.Path(pathlib.Path(__file__).parent, 'config/fd_schema.json'), 'r') as f:
            validate(instance={'experiment_definition': kwargs['experiment_definition']}, schema=json.load(f))
        return func(*args, **kwargs)

    return wrapper
