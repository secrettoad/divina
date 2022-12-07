import json
import os
import pathlib
from functools import wraps

import dask.dataframe as dd
import numpy as np
import s3fs
from dask_ml.linear_model import LinearRegression
from jsonschema import validate
from pipeline.model import GLM
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from dask.distributed import Client, LocalCluster
from kfp.components import create_component_from_func
from kubernetes.client.models import V1EnvVar
import collections
import dill
from typing import Union, List


# import kfp
# kfp.components._data_passing._converters.append(kfp.components._data_passing.Converter)


class Output(str):
    pass


class Input(str):
    pass


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


###TODO implement this as part of __eq__
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
            assert s[1].get_params(deep=True) == o[1].get_params(deep=True)
        if isinstance(s[0], GLM):
            assert np.allclose(s[1].linear_model.coef_, o[1].linear_model.coef_)
            assert np.allclose([s[1].linear_model.intercept_], [o[1].linear_model.intercept_])
        return None


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


def _component_helper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import inspect
        sig = inspect.signature(func)
        self = args[0]

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

        _args = dict(sig.parameters)
        new_args = []
        for a, v in zip(_args, args):
            if _args[a].annotation == Union[str, dd.DataFrame] and type(v) == str:
                new_args.append(dd.read_parquet(v, storage_options=self.storage_options))
            elif _args[a].annotation == Union[List[str], List[dd.DataFrame]] and type(v) == [str]:
                new_args.append([dd.read_parquet(_df, storage_options=self.storage_options) for _df in
                             v])
            elif _args[a].annotation == Union[str, BaseEstimator] and type(v) == str:
                fs = s3fs.S3FileSystem(**self.storage_options)
                with fs.open(v, 'rb') as f:
                    new_args.append(dill.load(f))

            else:
                new_args.append(v)
        new_args = tuple(new_args)
        result = func(*new_args, **kwargs)
        from itertools import zip_longest
        outputs = [v for v, a in zip_longest(args, _args) if _args[a].annotation == Output]
        reduce_result = False
        if not type(result) == tuple:
            result = (result,)
            reduce_result = True
        if not len(outputs) == len(result):
            raise ValueError(
                'The same number of Outputs must be designated in component signature as returned within component function')

        for o, r in zip(outputs, result):
            if type(r) == dd.DataFrame and o:
                npartitions = (r.memory_usage(deep=True).sum().compute() // 104857600) + 1
                r = cull_empty_partitions(r)
                r = r.repartition(npartitions=npartitions)
                r.to_parquet(o, storage_options=self.storage_options)
            if type(r) == dict and o:
                fs = s3fs.S3FileSystem(**self.storage_options)
                with fs.open(o + '.json', 'w') as f:
                    json.dump(r, f)
            if isinstance(r, BaseEstimator) and o:
                fs = s3fs.S3FileSystem(**self.storage_options)
                with fs.open(o + '.pkl', 'wb') as f:
                    dill.dump(r, f)

        if reduce_result:
            return result[0]
        else:
            return result

    return wrapper


class testClass:
    def __init__(self, value):
        self.value = value


from prefect import task

def generate_random_key(length):
    import random
    import string
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))

def _divina_component(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'prefect' in kwargs and kwargs['prefect']:
            kwargs.pop('prefect')
            _task = task(_component_helper(func))
            self = args[0]
            from inspect import signature
            _args = dict(signature(func).parameters)
            for a in _args:
                if _args[a].annotation == Output:
                    kwargs[a] = '/'.join([self.pipeline_root, func.__name__, generate_random_key(15), a])
            return _task(*args, **kwargs)
        else:
            if 'prefect' in kwargs:
                kwargs.pop('prefect')
            return _component_helper(func)(*args, **kwargs)
    return wrapper


def create_dask_aws_cluster(aws_workers, ec2_key, keep_alive):
    cluster = EC2Cluster(
        key_name=ec2_key,
        security=False,
        docker_image="jhurdle/divina:latest",
        env_vars={
            "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
            "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
            "AWS_DEFAULT_REGION": os.environ["AWS_DEFAULT_REGION"],
        },
        auto_shutdown=not keep_alive,
    )
    cluster.scale(aws_workers)
    return cluster


def get_dask_client(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not kwargs["dask_scheduler_ip"]:
            if not kwargs["create_cluster_destination"]:
                with LocalCluster() as cluster:
                    with Client(cluster) as client:
                        value = func(*args, **kwargs)
                        cluster.close()
                        cluster.shutdown()
                        return value
            elif kwargs["create_cluster_destination"] == "aws":
                with create_dask_aws_cluster(kwargs["num_workers"], kwargs["ssh_key"],
                                             keep_alive=kwargs["debug"]) as cluster:
                    with Client(cluster) as client:
                        value = func(*args, **kwargs)
                        cluster.close()
                        cluster.shutdown()
                        return value
            ###todo add gcp, azure

        else:
            with Client(kwargs["dask_scheduler_ip"]):
                return func(*args, **kwargs)

    return wrapper
