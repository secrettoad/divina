import click
from ..train import dask_train
from ..predict import dask_predict
from ..forecast import get_parameters, set_parameters
from ..validate import dask_validate
from ..aws.utils import create_divina_role
from dask_cloudprovider.aws import EC2Cluster
from dask.distributed import Client
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


def cli_train_vision(
        s3_fs,
        forecast_definition,
        write_path,
        ec2_keypair_name=None,
        keep_instances_alive=False,
        local=False,
        debug=False,
        dask_client=None,
        random_state=None
):
    if local:
        with Client():
            dask_train(
                s3_fs=s3_fs,
                forecast_definition=forecast_definition,
                write_path=write_path,
                random_seed=random_state
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
                            forecast_definition=forecast_definition,
                            write_path=write_path,
                            random_seed=random_state
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
                    forecast_definition=forecast_definition,
                    write_path=write_path,
                    random_seed=random_state
                )

    else:
        dask_train(
            s3_fs=s3_fs,
            forecast_definition=forecast_definition,
            write_path=write_path,
            random_seed=random_state
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
        dask_client=None
):
    if local:
        with Client():
            dask_predict(
                s3_fs=s3_fs,
                forecast_definition=forecast_definition,
                write_path=write_path,
                read_path=read_path
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
                            read_path=read_path
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
                    read_path=read_path
                )

    else:
        dask_predict(
            s3_fs=s3_fs,
            forecast_definition=forecast_definition,
            write_path=write_path,
            read_path=read_path
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
@click.argument("write_path", default="divina-forecast", required=False)
@click.argument("forecast_def", type=click.File("rb"))
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
    forecast_def = json.load(forecast_def)['forecast_definition']
    ###TODO create get_dask_client function that accepts ec2key and keep alive
    if not local:
        with EC2Cluster(
                key_name=ec2_key,
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
                cli_train_vision(
                    s3_fs=s3fs.S3FileSystem(),
                    forecast_definition=forecast_def,
                    write_path=write_path,
                    ec2_keypair_name=ec2_key,
                    keep_instances_alive=keep_alive,
                    local=local,
                    debug=debug,
                )
                cli_predict_vision(
                    s3_fs=s3fs.S3FileSystem(),
                    forecast_definition=forecast_def,
                    write_path=write_path,
                    read_path=write_path,
                    ec2_keypair_name=ec2_key,
                    keep_instances_alive=keep_alive,
                    local=local,
                    debug=debug,
                )
                cli_validate_vision(
                    s3_fs=s3fs.S3FileSystem(),
                    forecast_definition=forecast_def,
                    write_path=write_path,
                    read_path=write_path,
                    ec2_keypair_name=ec2_key,
                    keep_instances_alive=keep_alive,
                    local=local,
                    debug=debug,
                )
    else:
        cli_train_vision(
            s3_fs=s3fs.S3FileSystem(),
            forecast_definition=forecast_def,
            write_path=write_path,
            ec2_keypair_name=ec2_key,
            keep_instances_alive=keep_alive,
            local=local,
            debug=debug,
        )
        cli_predict_vision(
            s3_fs=s3fs.S3FileSystem(),
            forecast_definition=forecast_def,
            write_path=write_path,
            read_path=write_path,
            ec2_keypair_name=ec2_key,
            keep_instances_alive=keep_alive,
            local=local,
            debug=debug,
        )
        cli_validate_vision(
            s3_fs=s3fs.S3FileSystem(),
            forecast_definition=forecast_def,
            write_path=write_path,
            read_path=write_path,
            ec2_keypair_name=ec2_key,
            keep_instances_alive=keep_alive,
            local=local,
            debug=debug,
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
