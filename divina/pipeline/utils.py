import json
import os
import pathlib
from functools import wraps
from typing import List, Union

import dask.dataframe as dd
import dill
import s3fs
from dask.distributed import Client, LocalCluster
from dask_cloudprovider.aws import EC2Cluster
from prefect import task
from sklearn.base import BaseEstimator


class Output(str):
    pass


class Input(str):
    pass


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


def _component_helper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import inspect

        sig = inspect.signature(func)
        self = args[0]

        _args = dict(sig.parameters)
        new_args = []
        for a, v in zip(_args, args):
            if (_args[a].annotation == Union[str, dd.DataFrame]) and type(v) == str:
                new_args.append(
                    dd.read_parquet(v, storage_options=self.storage_options)
                )
            elif (_args[a].annotation == Union[str, dd.Series]) and type(v) == str:
                s = dd.read_parquet(v, storage_options=self.storage_options)
                if len(s.columns) > 1:
                    raise ValueError(
                        "Dask Series expected but "
                        "multi-column dataframe encountered."
                    )
                s = s[s.columns[0]]
                new_args.append(s)
            elif _args[a].annotation == Union[List[str], List[dd.DataFrame]] and type(
                v
            ) == [str]:
                new_args.append(
                    [
                        dd.read_parquet(_df, storage_options=self.storage_options)
                        for _df in v
                    ]
                )
            elif _args[a].annotation == Union[str, BaseEstimator] and type(v) == str:
                fs = s3fs.S3FileSystem(**self.storage_options)
                with fs.open(v, "rb") as f:
                    new_args.append(dill.load(f))

            else:
                new_args.append(v)
        new_args = tuple(new_args)
        result = func(*new_args, **kwargs)
        from itertools import zip_longest

        outputs = [
            v for v, a in zip_longest(args, _args) if _args[a].annotation == Output
        ]
        reduce_result = False
        if not type(result) == tuple:
            result = (result,)
            reduce_result = True
        if not len(outputs) == len(result):
            raise ValueError(
                "The same number of Outputs must be "
                "designated in component signature as "
                "returned within component function"
            )

        for o, r in zip(outputs, result):
            if (type(r) == dd.Series or type(r) == dd.DataFrame) and o:
                r = r.repartition(partition_size=104857600)
                if type(r) == dd.DataFrame:
                    r.to_parquet(o, storage_options=self.storage_options)
                elif type(r) == dd.Series:
                    r.to_frame().to_parquet(o, storage_options=self.storage_options)
            if type(r) == dict and o:
                fs = s3fs.S3FileSystem(**self.storage_options)
                with fs.open(o + ".json", "w") as f:
                    json.dump(r, f)
            if isinstance(r, BaseEstimator) and o:
                fs = s3fs.S3FileSystem(**self.storage_options)
                with fs.open(o + ".pkl", "wb") as f:
                    dill.dump(r, f)

        if reduce_result:
            return result[0]
        else:
            return result

    return wrapper


class testClass:
    def __init__(self, value):
        self.value = value


def generate_random_key(length):
    import random
    import string

    return "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(length)
    )


def _divina_component(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "prefect" in kwargs and kwargs["prefect"]:
            kwargs.pop("prefect")
            _task = task(_component_helper(func))
            self = args[0]
            from inspect import signature

            _args = dict(signature(func).parameters)
            for a in _args:
                if _args[a].annotation == Output:
                    if self.pipeline_root:
                        kwargs[a] = "/".join(
                            [
                                self.pipeline_root,
                                func.__name__,
                                generate_random_key(15),
                                a,
                            ]
                        )
                    else:
                        kwargs[a] = None
            return _task(*args, **kwargs)
        else:
            if "prefect" in kwargs:
                kwargs.pop("prefect")
            return _component_helper(func)(*args, **kwargs)

    return wrapper


def create_dask_aws_cluster(
    num_workers, ec2_key=None, keep_alive=False, docker_image=None
):
    region = (
        "us-east-1"
        if "AWS_DEFAULT_REGION" not in os.environ
        else os.environ["AWS_DEFAULT_REGION"]
    )
    cluster = EC2Cluster(
        key_name=ec2_key,
        security=False,
        # TODO - START HERE - add bokeh to docker image then check to see if versions match and error is resolved
        docker_image=docker_image or "jhurdle/divina:latest",
        env_vars={
            "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
            "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
            "AWS_DEFAULT_REGION": region,
        },
        auto_shutdown=not keep_alive,
        region=region,
    )
    cluster.scale(num_workers)
    return cluster


class DaskConfiguration:
    def __init__(
        self,
        scheduler_ip=None,
        destination=None,
        num_workers=None,
        ssh_key=None,
        debug=False,
        docker_image=None,
    ):
        if not scheduler_ip and not destination:
            self.destination = "local"
        elif scheduler_ip and destination:
            raise ValueError("Either scheduler_ip or destination can be set, not both")
        else:
            self.scheduler_ip = scheduler_ip
            self.destination = destination
        self.num_workers = num_workers
        self.ssh_key = ssh_key
        self.debug = debug
        self.docker_image = docker_image


def get_dask_client(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "dask_configuration" not in kwargs:
            kwargs["dask_configuration"] = None
            return func(*args, **kwargs)
        else:
            config = kwargs["dask_configuration"]
        if hasattr(config, "dask_scheduler_ip"):
            with Client(config.dask_scheduler_ip):
                return func(*args, **kwargs)
        elif config.destination == "aws":
            with create_dask_aws_cluster(
                num_workers=config.num_workers,
                ec2_key=config.ssh_key,
                keep_alive=config.debug,
                docker_image=config.docker_image,
            ) as cluster:
                with Client(cluster) as client:  # noqa: F841
                    value = func(*args, **kwargs)
                    cluster.close()
                    client.shutdown()
                    return value
            # todo add gcp, azure

    return wrapper
