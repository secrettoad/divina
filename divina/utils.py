import json
import os
import pathlib
from functools import wraps

import dask.dataframe as dd
import numpy as np
import s3fs
from dask.distributed import Client
from dask_cloudprovider.aws import EC2Cluster
from dask_ml.linear_model import LinearRegression
from jsonschema import validate
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


def get_parameters(model_path):
    if model_path[:5] == "s3://":
        s3_fs = s3fs.S3FileSystem()
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
    with write_open("{}_params".format(model_path), "rb") as f:
        params = json.load(f)
        return params


def set_parameters(model_path, params):
    if model_path[:5] == "s3://":
        s3_fs = s3fs.S3FileSystem()
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
    with write_open("{}_params".format(model_path), "rb") as f:
        parameters = json.load(f)["features"]
    if not params == parameters:
        raise Exception(
            "Parameters {} not found in trained model. Cannot set new values for these parameters".format(
                ", ".join(list(set(params) - set(parameters)))
            )
        )
    else:
        for p in params:
            if not p in parameters:
                parameters.append(p)
        with write_open("{}_params".format(model_path), "w") as f:
            json.dump({"features": parameters}, f)


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
        with open(
            pathlib.Path(pathlib.Path(__file__).parent, "config/ed_schema.json"), "r"
        ) as f:
            validate(
                instance={"experiment_definition": kwargs["experiment_definition"]},
                schema=json.load(f),
            )
        return func(*args, **kwargs)

    return wrapper


def create_write_directory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        write_path = kwargs["write_path"]
        s3_fs = s3fs.S3FileSystem()
        if write_path[:5] == "s3://":
            if not s3_fs.exists(write_path):
                s3_fs.mkdir(
                    write_path,
                    create_parents=True,
                    region_name=os.environ["AWS_DEFAULT_REGION"],
                    acl="private",
                )
        else:
            path = pathlib.Path(write_path)
            path.mkdir(exist_ok=True, parents=True)
            pathlib.Path(path, "models/bootstrap").mkdir(exist_ok=True, parents=True)
            pathlib.Path(path, "models/validation").mkdir(exist_ok=True, parents=True)
        return func(*args, **kwargs)

    return wrapper


def get_dask_client(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not "aws_workers" in kwargs:
            with Client():
                return func(*args, **kwargs)
        else:
            aws_workers = kwargs["aws_workers"]
            if not "ec2_key" in kwargs:
                ec2_key = None
            else:
                ec2_key = kwargs["ec2_key"]
            if not "keep_alive" in kwargs:
                keep_alive = False
            else:
                keep_alive = kwargs["keep_alive"]
            with EC2Cluster(
                key_name=ec2_key,
                security=False,
                docker_image="jhurdle/divina:latest",
                env_vars={
                    "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                    "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                    "AWS_DEFAULT_REGION": os.environ["AWS_DEFAULT_REGION"],
                },
                auto_shutdown=not keep_alive,
            ) as cluster:
                cluster.scale(aws_workers)
                with Client(cluster) as client:
                    return func(*args, **kwargs)

    return wrapper
