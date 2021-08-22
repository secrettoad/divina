import click
from ..dataset import build_dataset_dask
from ..train import dask_train
from ..predict import dask_predict
from ..validate import dask_validate
from dask_cloudprovider.aws import EC2Cluster
from dask.distributed import Client
from dask_ml.linear_model import LinearRegression
from botocore.exceptions import NoRegionError
from botocore.exceptions import NoCredentialsError
import os
import sys
import json


def cli_build_dataset(
    read_path,
    write_path,
    dataset_name,
    ec2_keypair_name=None,
    keep_instances_alive=False,
    local=False,
    debug=False,
    dask_address=None,
):

    if local:
        with Client():
            build_dataset_dask(
                read_path=read_path, write_path=write_path, dataset_name=dataset_name
            )
    elif not dask_address:
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
                ) as cluster:
                    cluster.adapt(minimum=0, maximum=10)
                    with Client(cluster):
                        build_dataset_dask(
                            read_path=read_path,
                            write_path=write_path,
                            dataset_name=dataset_name,
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
                    read_path=read_path,
                    write_path=write_path,
                    dataset_name=dataset_name,
                )

    else:
        with Client(dask_address):
            build_dataset_dask(
                read_path=read_path, write_path=write_path, dataset_name=dataset_name
            )


def cli_train_vision(
    vision_definition,
    write_path,
    vision_name,
    ec2_keypair_name=None,
    keep_instances_alive=False,
    local=False,
    debug=False,
    dask_address=None,
):
    dask_model = LinearRegression
    if local:
        with Client() as dask_client:
            dask_train(
                dask_client=dask_client,
                dask_model=dask_model,
                vision_definition=vision_definition,
                divina_directory=write_path,
                vision_id=vision_name,
            )
    elif not dask_address:
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
                ) as cluster:
                    cluster.adapt(minimum=0, maximum=10)
                    with Client(cluster) as dask_client:
                        dask_train(
                            dask_client=dask_client,
                            dask_model=dask_model,
                            vision_definition=vision_definition,
                            divina_directory=write_path,
                            vision_id=vision_name,
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
            with Client(cluster) as dask_client:
                dask_train(
                    dask_client=dask_client,
                    dask_model=dask_model,
                    vision_definition=vision_definition,
                    divina_directory=write_path,
                    vision_id=vision_name,
                )

    else:
        with Client(dask_address) as dask_client:
            dask_train(
                dask_client=dask_client,
                dask_model=dask_model,
                vision_definition=vision_definition,
                divina_directory=write_path,
                vision_id=vision_name,
            )


def cli_predict_vision(
    s3_fs,
    vision_definition,
    write_path,
    vision_name,
    ec2_keypair_name=None,
    keep_instances_alive=False,
    local=False,
    debug=False,
    dask_address=None,
):
    if local:
        with Client() as dask_client:
            dask_predict(
                s3_fs=s3_fs,
                dask_client=dask_client,
                vision_definition=vision_definition,
                divina_directory=write_path,
                vision_id=vision_name,
            )
    elif not dask_address:
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
                ) as cluster:
                    cluster.adapt(minimum=0, maximum=10)
                    with Client(cluster) as dask_client:
                        dask_predict(
                            s3_fs=s3_fs,
                            dask_client=dask_client,
                            vision_definition=vision_definition,
                            divina_directory=write_path,
                            vision_id=vision_name,
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
            with Client(cluster) as dask_client:
                dask_predict(
                    s3_fs=s3_fs,
                    dask_client=dask_client,
                    vision_definition=vision_definition,
                    divina_directory=write_path,
                    vision_id=vision_name,
                )

    else:
        with Client(dask_address) as dask_client:
            dask_predict(
                s3_fs=s3_fs,
                dask_client=dask_client,
                vision_definition=vision_definition,
                divina_directory=write_path,
                vision_id=vision_name,
            )


def cli_validate_vision(
    s3_fs,
    vision_definition,
    write_path,
    vision_name,
    ec2_keypair_name=None,
    keep_instances_alive=False,
    local=False,
    debug=False,
    dask_address=None,
):
    if local:
        with Client() as dask_client:
            dask_validate(
                s3_fs=s3_fs,
                dask_client=dask_client,
                vision_definition=vision_definition,
                divina_directory=write_path,
                vision_id=vision_name,
            )
    elif not dask_address:
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
                ) as cluster:
                    cluster.adapt(minimum=0, maximum=10)
                    with Client(cluster) as dask_client:
                        dask_validate(
                            s3_fs=s3_fs,
                            dask_client=dask_client,
                            vision_definition=vision_definition,
                            divina_directory=write_path,
                            vision_id=vision_name,
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
            with Client(cluster) as dask_client:
                dask_validate(
                    s3_fs=s3_fs,
                    dask_client=dask_client,
                    vision_definition=vision_definition,
                    divina_directory=write_path,
                    vision_id=vision_name,
                )

    else:
        with Client(dask_address) as dask_client:
            dask_validate(
                s3_fs=s3_fs,
                dask_client=dask_client,
                vision_definition=vision_definition,
                divina_directory=write_path,
                vision_id=vision_name,
            )


@click.group()
def divina():
    pass


@divina.group()
def dataset():
    pass


@divina.group()
def vision():
    pass


@click.argument("ec2_keypair_name", default=None, required=False)
@click.argument("keep_instances_alive", default=False, required=False)
@click.argument("dataset_name")
@click.argument("write_path")
@click.argument("read_path")
@click.option("-l", "--local", is_flag=True)
@dataset.command()
def build(
    dataset_name,
    write_path,
    read_path,
    ec2_keypair_name,
    keep_instances_alive,
    local,
):
    if not read_path[:5] == "s3://" and write_path[:5] == "s3://":
        raise Exception("both read_path and write_path must begin with 's3://'")
    cli_build_dataset(
        dataset_name=dataset_name,
        write_path=write_path,
        read_path=read_path,
        ec2_keypair_name=ec2_keypair_name,
        keep_instances_alive=keep_instances_alive,
        local=local,
    )


@click.argument("ec2_keypair_name", default=None, required=False)
@click.argument("keep_instances_alive", default=False, required=False)
@click.argument("vision_definition", type=click.File("rb"))
@click.argument("vision_name")
@click.argument("write_path")
@click.option("-l", "--local", is_flag=True)
@click.option("-d", "--debug", is_flag=True)
@vision.command()
def train(
    vision_definition,
    vision_name,
    keep_instances_alive,
    ec2_keypair_name,
    write_path,
    local,
    debug,
):
    cli_train_vision(
        vision_definition=json.load(vision_definition),
        write_path=write_path,
        vision_name=vision_name,
        ec2_keypair_name=ec2_keypair_name,
        keep_instances_alive=keep_instances_alive,
        local=local,
        debug=debug,
    )


@click.argument("s3_endpoint")
@click.argument("data_definition", type=click.File("rb"))
@divina.command()
def predict(s3_endpoint, data_definition):
    sc = get_spark_context_s3(s3_endpoint)
    predict.predict(spark_context=sc, data_definition=data_definition)


@click.argument("s3_endpoint")
@click.argument("data_definition", type=click.File("rb"))
@divina.command()
def validate(s3_endpoint, data_definition):
    sc = get_spark_context_s3(s3_endpoint)
    validate.validate(spark_context=sc, data_definition=data_definition)
