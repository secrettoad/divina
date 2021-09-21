import click
from ..dataset import build_dataset_dask
from ..train import dask_train
from ..predict import dask_predict
from ..vision import get_parameters, set_parameters
from ..validate import dask_validate
from ..aws.utils import create_divina_role
from dask_cloudprovider.aws import EC2Cluster
from dask.distributed import Client
from dask_ml.linear_model import LinearRegression
from botocore.exceptions import NoRegionError
from botocore.exceptions import NoCredentialsError
import os
import sys
import json
import boto3
import s3fs


def upsert_divina_iam():
    divina_session = boto3.session.Session()
    role, instance_profile = create_divina_role(divina_session)
    return role, instance_profile


def cli_get_params(
        model_path,
        s3_fs
):
    return get_parameters(s3_fs=s3_fs, model_path=model_path)


def cli_set_params(
        model_path,
        s3_fs,
        params
):
    return set_parameters(s3_fs=s3_fs, model_path=model_path, params=params)


def cli_build_dataset(
    read_path,
    write_path,
    s3_fs,
    ec2_keypair_name=None,
    keep_instances_alive=False,
    local=False,
    debug=False,
    dask_client=None,
):
    if local:
        with Client():
            build_dataset_dask(
                s3_fs=s3_fs,
                read_path=read_path,
                write_path=write_path,
            )
    elif not dask_client:
        if not keep_instances_alive:
            try:
                with EC2Cluster(
                    key_name=ec2_keypair_name,
                    security=False,
                    docker_image="jhurdle/divina:latest",
                    debug=debug,
                    env_vars={
                        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                    },
                    auto_shutdown=True,
                ) as cluster:
                    cluster.adapt(minimum=0, maximum=10)
                    with Client(cluster):
                        build_dataset_dask(
                            s3_fs=s3_fs,
                            read_path=read_path,
                            write_path=write_path,
                        )
            except NoRegionError:
                sys.stderr.write(
                    "No AWS region configured. Please set AWS_DEFAULT_REGION environment variable."
                )
            except NoCredentialsError:
                sys.stderr.write(
                    "No AWS credentials configured. Please set AWS_SECRET_ACCESS_KEY and AWS_ACCESS_KEY_ID environment variables."
                )
        else:
            cluster = EC2Cluster(
                key_name=ec2_keypair_name,
                security=False,
                debug=debug,
                env_vars={
                    "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                    "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                },
            )
            cluster.adapt(minimum=0, maximum=10)
            with Client(cluster):
                build_dataset_dask(
                    s3_fs=s3_fs, read_path=read_path, write_path=write_path
                )

    else:
        try:
            build_dataset_dask(s3_fs=s3_fs, read_path=read_path, write_path=write_path)
        except:
            pass


def cli_train_vision(
    s3_fs,
    forecast_definition,
    write_path,
    ec2_keypair_name=None,
    keep_instances_alive=False,
    local=False,
    debug=False,
    dask_client=None,
):

    dask_model = LinearRegression
    if local:
        with Client():
            dask_train(
                s3_fs=s3_fs,
                dask_model=dask_model,
                forecast_definition=forecast_definition,
                write_path=write_path,
            )
    elif not dask_client:
        if not keep_instances_alive:
            try:
                with EC2Cluster(
                    key_name=ec2_keypair_name,
                    security=False,
                    docker_image="jhurdle/divina:latest",
                    debug=debug,
                    env_vars={
                        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                    },
                    auto_shutdown=True,
                ) as cluster:
                    cluster.adapt(minimum=0, maximum=10)
                    with Client(cluster):
                        dask_train(
                            s3_fs=s3_fs,
                            dask_model=dask_model,
                            forecast_definition=forecast_definition,
                            write_path=write_path,
                        )
            except NoRegionError:
                sys.stderr.write(
                    "No AWS region configured. Please set AWS_DEFAULT_REGION environment variable."
                )
            except NoCredentialsError:
                sys.stderr.write(
                    "No AWS credentials configured. Please set AWS_SECRET_ACCESS_KEY and AWS_ACCESS_KEY_ID environment variables."
                )
        else:
            cluster = EC2Cluster(
                key_name=ec2_keypair_name,
                security=False,
                debug=debug,
                env_vars={
                    "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                    "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                },
            )
            cluster.adapt(minimum=0, maximum=10)
            with Client(cluster):
                dask_train(
                    s3_fs=s3_fs,
                    dask_model=dask_model,
                    forecast_definition=forecast_definition,
                    write_path=write_path,
                )

    else:
        dask_train(
            s3_fs=s3_fs,
            dask_model=dask_model,
            forecast_definition=forecast_definition,
            write_path=write_path,
        )


def cli_predict_vision(
    s3_fs,
    forecast_definition,
    write_path,
    read_path,
    ec2_keypair_name=None,
    keep_instances_alive=False,
    local=False,
    debug=False,
    dask_client=None,
):
    if local:
        with Client():
            dask_predict(
                s3_fs=s3_fs,
                forecast_definition=forecast_definition,
                write_path=write_path,
                read_path=read_path,
            )
    elif not dask_client:
        if not keep_instances_alive:
            try:
                with EC2Cluster(
                    key_name=ec2_keypair_name,
                    security=False,
                    docker_image="jhurdle/divina:latest",
                    debug=debug,
                    env_vars={
                        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                    },
                    auto_shutdown=True,
                ) as cluster:
                    cluster.adapt(minimum=0, maximum=10)
                    with Client(cluster):
                        dask_predict(
                            s3_fs=s3_fs,
                            forecast_definition=forecast_definition,
                            write_path=write_path,
                            read_path=read_path,
                        )
            except NoRegionError:
                sys.stderr.write(
                    "No AWS region configured. Please set AWS_DEFAULT_REGION environment variable."
                )
            except NoCredentialsError:
                sys.stderr.write(
                    "No AWS credentials configured. Please set AWS_SECRET_ACCESS_KEY and AWS_ACCESS_KEY_ID environment variables."
                )
        else:
            cluster = EC2Cluster(
                key_name=ec2_keypair_name,
                security=False,
                debug=debug,
                env_vars={
                    "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                    "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                },
            )
            cluster.adapt(minimum=0, maximum=10)
            with Client(cluster):
                dask_predict(
                    s3_fs=s3_fs,
                    forecast_definition=forecast_definition,
                    write_path=write_path,
                    read_path=read_path,
                )

    else:
        dask_predict(
            s3_fs=s3_fs,
            forecast_definition=forecast_definition,
            write_path=write_path,
            read_path=read_path,
        )


def cli_validate_vision(
    s3_fs,
    forecast_definition,
    write_path,
    read_path,
    ec2_keypair_name=None,
    keep_instances_alive=False,
    local=False,
    debug=False,
    dask_client=None,
):
    if local:
        with Client():
            dask_validate(
                s3_fs=s3_fs,
                forecast_definition=forecast_definition,
                write_path=write_path,
                read_path=read_path,
            )
    elif not dask_client:
        if not keep_instances_alive:
            try:
                with EC2Cluster(
                    key_name=ec2_keypair_name,
                    security=False,
                    docker_image="jhurdle/divina:latest",
                    debug=debug,
                    env_vars={
                        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                    },
                    auto_shutdown=True,
                ) as cluster:
                    cluster.adapt(minimum=0, maximum=10)
                    with Client(cluster):
                        dask_validate(
                            s3_fs=s3_fs,
                            forecast_definition=forecast_definition,
                            write_path=write_path,
                            read_path=read_path,
                        )
            except NoRegionError:
                sys.stderr.write(
                    "No AWS region configured. Please set AWS_DEFAULT_REGION environment variable."
                )
            except NoCredentialsError:
                sys.stderr.write(
                    "No AWS credentials configured. Please set AWS_SECRET_ACCESS_KEY and AWS_ACCESS_KEY_ID environment variables."
                )
        else:
            cluster = EC2Cluster(
                key_name=ec2_keypair_name,
                security=False,
                debug=debug,
                env_vars={
                    "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                    "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                },
            )
            cluster.adapt(minimum=0, maximum=10)
            with Client(cluster):
                dask_validate(
                    s3_fs=s3_fs,
                    forecast_definition=forecast_definition,
                    write_path=write_path,
                    read_path=read_path,
                )

    else:
        dask_validate(
            s3_fs=s3_fs,
            forecast_definition=forecast_definition,
            write_path=write_path,
            read_path=read_path,
        )


@click.group()
def divina():
    pass


@click.argument(
    "ec2_key",
    default=None,
    required=False,
)
@click.argument("keep_alive", default=False, required=False)
@click.argument("vision_def", type=click.File("rb"))
@click.argument("write_path", default="divina-forecast", required=False)
@click.option("-l", "--local", is_flag=True, help="flag to compute results locally")
@click.option(
    "-d", "--debug", is_flag=True, help="flag to increase verbosity of console output"
)
@divina.command()
def forecast(forecast_def, keep_alive, ec2_key, write_path, local, debug):
    """:write_path: s3:// or local path to write results to
    :forecast_def: path to vision definition JSON file
    :keep_alive: flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging.
    :ec2_key: aws ec2 keypair name to provide access to dask cluster for debugging.
    """
    cli_train_vision(
        s3_fs=s3fs.S3FileSystem(),
        forecast_definition=json.load(forecast_def),
        write_path=write_path,
        ec2_keypair_name=ec2_key,
        keep_instances_alive=keep_alive,
        local=local,
        debug=debug,
    )
    cli_predict_vision(
        s3_fs=s3fs.S3FileSystem(),
        forecast_definition=json.load(forecast_def),
        write_path=write_path,
        read_path=write_path,
        ec2_keypair_name=ec2_key,
        keep_instances_alive=keep_alive,
        local=local,
        debug=debug,
    )
    cli_validate_vision(
        s3_fs=s3fs.S3FileSystem(),
        forecast_definition=json.load(forecast_def),
        write_path=write_path,
        read_path=write_path,
        ec2_keypair_name=ec2_key,
        keep_instances_alive=keep_alive,
        local=local,
        debug=debug,
    )


@click.argument("ec2_key", default=None, required=False)
@click.argument("keep_alive", default=False, required=False)
@click.argument("write_path", default="divina-forecast", required=False)
@click.argument("read_path", default="divina-forecast", required=False)
@click.option("-l", "--local", is_flag=True, help="flag to compute results locally")
@divina.command()
def dataset(
    write_path,
    read_path,
    ec2_key,
    keep_alive,
    local,
):
    """:read_path: s3:// or local path to read raw data from
    :write_path: s3:// or local path to write results to
    :keep_alive: flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging
    :ec2_key: aws ec2 keypair name to provide access to dask cluster for debugging
    """
    if not read_path[:5] == "s3://" and write_path[:5] == "s3://":
        raise Exception("both read_path and write_path must begin with 's3://'")
    cli_build_dataset(
        s3_fs=s3fs.S3FileSystem(),
        write_path=write_path,
        read_path=read_path,
        ec2_keypair_name=ec2_key,
        keep_instances_alive=keep_alive,
        local=local,
    )


@click.argument("ec2_keypair_name", default=None, required=False)
@click.argument("keep_instances_alive", default=False, required=False)
@click.argument("forecast_definition", type=click.File("rb"))
@click.argument("write_path", default="divina-forecast", required=False)
@click.option("-l", "--local", is_flag=True, help="flag to compute results locally")
@click.option(
    "-d", "--debug", is_flag=True, help="flag to increase verbosity of console output"
)
@divina.command()
def train(
    forecast_def,
    keep_alive,
    ec2_key,
    write_path,
    local,
    debug,
):
    """:write_path: s3:// or local path to write trained model to
    :forecast_def: path to vision definition JSON file
    :keep_alive: flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging
    :ec2_key: aws ec2 keypair name to provide access to dask cluster for debugging
    """
    cli_train_vision(
        s3_fs=s3fs.S3FileSystem(),
        forecast_definition=json.load(forecast_def),
        write_path=write_path,
        ec2_keypair_name=ec2_key,
        keep_instances_alive=keep_alive,
        local=local,
        debug=debug,
    )


@click.argument("ec2_key", default=None, required=False)
@click.argument("keep_alive", default=False, required=False)
@click.argument("vision_def", type=click.File("rb"))
@click.argument("write_path", default="divina-forecast", required=False)
@click.argument("read_path", default="divina-forecast", required=False)
@click.option("-l", "--local", is_flag=True, help="flag to compute results locally")
@click.option(
    "-d", "--debug", is_flag=True, help="flag to increase verbosity of console output"
)
@divina.command()
def predict(
    forecast_def,
    keep_alive,
    ec2_key,
    write_path,
    read_path,
    local,
    debug,
):
    """:read_path: s3:// or local path to read trained model fromn
    :write_path: s3:// or local path to write results to
    :forecast_def: path to vision definition JSON file
    :keep_alive: flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging
    :ec2_key: aws ec2 keypair name to provide access to dask cluster for debugging
    """
    cli_predict_vision(
        s3_fs=s3fs.S3FileSystem(),
        forecast_definition=json.load(forecast_def),
        write_path=write_path,
        read_path=read_path,
        ec2_keypair_name=ec2_key,
        keep_instances_alive=keep_alive,
        local=local,
        debug=debug,
    )


@click.argument(
    "ec2_key",
    default=None,
    required=False,
)
@click.argument("keep_alive", default=False, required=False)
@click.argument("vision_def", type=click.File("rb"))
@click.argument("write_path", default="divina-forecast", required=False)
@click.argument("read_path", default="divina-forecast", required=False)
@click.option("-l", "--local", is_flag=True, help="flag to compute results locally")
@click.option(
    "-d", "--debug", is_flag=True, help="flag to increase verbosity of console output"
)
@divina.command()
def validate(
    forecast_def,
    keep_alive,
    ec2_key,
    write_path,
    read_path,
    local,
    debug,
):
    """:read_path: s3:// or local path to read predictions from
    :write_path: s3:// or local path to write results to
    :forecast_def: path to vision definition JSON file
    :keep_alive: flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging
    :ec2_key: aws ec2 keypair name to provide access to dask cluster for debugging
    """
    cli_validate_vision(
        s3_fs=s3fs.S3FileSystem(),
        forecast_definition=json.load(forecast_def),
        write_path=write_path,
        read_path=read_path,
        ec2_keypair_name=ec2_key,
        keep_instances_alive=keep_alive,
        local=local,
        debug=debug,
    )


@click.argument("model_path", required=True)
@divina.command()
def get_params(
    model_path
):
    """:model_path: s3:// or local path to model to get parameters from
    """
    cli_get_params(
        s3_fs=s3fs.S3FileSystem(),
        model_path=model_path
    )


@click.argument("model_path", required=True)
@divina.command()
def set_params(
    model_path,
    params
):
    """:model_path: s3:// or local path to model to get parameters from
    :params: dictionary of trained model parameters to update
    """
    cli_set_params(
        s3_fs=s3fs.S3FileSystem(),
        model_path=model_path,
        params=params
    )

