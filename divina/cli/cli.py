import click
from ..dataset import build_dataset_dask
from ..train import dask_train
from ..predict import dask_predict
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


def cli_build_dataset(
        read_path,
        write_path,
        dataset_name,
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
                dataset_name=dataset_name,
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
                ) as cluster:
                    cluster.adapt(minimum=0, maximum=10)
                    with Client(cluster):
                        build_dataset_dask(
                            s3_fs=s3_fs,
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
                    s3_fs=s3_fs,
                    read_path=read_path,
                    write_path=write_path,
                    dataset_name=dataset_name,
                )

    else:
        try:
            build_dataset_dask(
                s3_fs=s3_fs,
                read_path=read_path,
                write_path=write_path,
                dataset_name=dataset_name,
            )
        except:
            pass


def cli_train_vision(
        s3_fs,
        vision_definition,
        write_path,
        vision_name,
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
                vision_definition=vision_definition,
                write_path=write_path,
                vision_id=vision_name,
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
                ) as cluster:
                    cluster.adapt(minimum=0, maximum=10)
                    with Client(cluster):
                        dask_train(
                            s3_fs=s3_fs,
                            dask_model=dask_model,
                            vision_definition=vision_definition,
                            write_path=write_path,
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
            with Client(cluster):
                dask_train(
                    s3_fs=s3_fs,
                    dask_model=dask_model,
                    vision_definition=vision_definition,
                    write_path=write_path,
                    vision_id=vision_name,
                )

    else:
        dask_train(
            s3_fs=s3_fs,
            dask_model=dask_model,
            vision_definition=vision_definition,
            write_path=write_path,
            vision_id=vision_name,
        )


def cli_predict_vision(
        s3_fs,
        vision_definition,
        write_path,
        read_path,
        vision_name,
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
                vision_definition=vision_definition,
                write_path=write_path,
                read_path=read_path,
                vision_id=vision_name,
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
                ) as cluster:
                    cluster.adapt(minimum=0, maximum=10)
                    with Client(cluster):
                        dask_predict(
                            s3_fs=s3_fs,
                            vision_definition=vision_definition,
                            write_path=write_path,
                            read_path=read_path,
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
            with Client(cluster):
                dask_predict(
                    s3_fs=s3_fs,
                    vision_definition=vision_definition,
                    write_path=write_path,
                    read_path=read_path,
                    vision_id=vision_name,
                )

    else:
        dask_predict(
            s3_fs=s3_fs,
            vision_definition=vision_definition,
            write_path=write_path,
            read_path=read_path,
            vision_id=vision_name,
        )


def cli_validate_vision(
        s3_fs,
        vision_definition,
        write_path,
        read_path,
        vision_name,
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
                vision_definition=vision_definition,
                write_path=write_path,
                read_path=read_path,
                vision_id=vision_name,
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
                ) as cluster:
                    cluster.adapt(minimum=0, maximum=10)
                    with Client(cluster):
                        dask_validate(
                            s3_fs=s3_fs,
                            vision_definition=vision_definition,
                            write_path=write_path,
                            read_path=read_path,
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
            with Client(cluster):
                dask_validate(
                    s3_fs=s3_fs,
                    vision_definition=vision_definition,
                    write_path=write_path,
                    read_path=read_path,
                    vision_id=vision_name,
                )

    else:
        dask_validate(
            s3_fs=s3_fs,
            vision_definition=vision_definition,
            write_path=write_path,
            read_path=read_path,
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


@click.argument("ec2_keypair_name", default=None, required=False,
                help="aws ec2 keypair name to provide access to dask cluster for debugging.")
@click.argument("keep_instances_alive", default=False, required=False,
                help="flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging.")
@click.argument("dataset_name", help="name of dataset. must be unique to the set of datasets in write_path")
@click.argument("write_path", help="s3:// or local path to write results to")
@click.argument("read_path", help="s3:// or local path to read predictions from")
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
        s3_fs=s3fs.S3FileSystem(),
        dataset_name=dataset_name,
        write_path=write_path,
        read_path=read_path,
        ec2_keypair_name=ec2_keypair_name,
        keep_instances_alive=keep_instances_alive,
        local=local,
    )


@click.argument("ec2_keypair_name", default=None, required=False,
                help="aws ec2 keypair name to provide access to dask cluster for debugging.")
@click.argument("keep_instances_alive", default=False, required=False,
                help="flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging.")
@click.argument("vision_definition", type=click.File("rb"), help="path to vision definition json file")
@click.argument("vision_name", help="path to vision definition json file")
@click.argument("write_path", help="s3:// or local path to write results to")
@click.option("-l", "--local", is_flag=True, "flag to compute results locally")
@click.option("-d", "--debug", is_flag=True, "flag to increase verbosity of console output")
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
        s3_fs=s3fs.S3FileSystem(),
        vision_definition=json.load(vision_definition),
        write_path=write_path,
        vision_name=vision_name,
        ec2_keypair_name=ec2_keypair_name,
        keep_instances_alive=keep_instances_alive,
        local=local,
        debug=debug,
    )


@click.argument("ec2_keypair_name", default=None, required=False,
                help="aws ec2 keypair name to provide access to dask cluster for debugging.")
@click.argument("keep_instances_alive", default=False, required=False,
                help="flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging.")
@click.argument("vision_definition", type=click.File("rb"), help="path to vision definition json file")
@click.argument("vision_name", help="path to vision definition json file")
@click.argument("write_path", help="s3:// or local path to write results to")
@click.argument("read_path", help="s3:// or local path to read predictions from")
@click.option("-l", "--local", is_flag=True, "flag to compute results locally")
@click.option("-d", "--debug", is_flag=True, "flag to increase verbosity of console output")
@vision.command()
def predict(
        vision_definition,
        vision_name,
        keep_instances_alive,
        ec2_keypair_name,
        write_path,
        read_path,
        local,
        debug,
):
    cli_predict_vision(
        s3_fs=s3fs.S3FileSystem(),
        vision_definition=json.load(vision_definition),
        write_path=write_path,
        read_path=read_path,
        vision_name=vision_name,
        ec2_keypair_name=ec2_keypair_name,
        keep_instances_alive=keep_instances_alive,
        local=local,
        debug=debug,
    )


@click.argument("ec2_keypair_name", default=None, required=False,
                help="aws ec2 keypair name to provide access to dask cluster for debugging.")
@click.argument("keep_instances_alive", default=False, required=False,
                help="flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging.")
@click.argument("vision_definition", type=click.File("rb"), help="path to vision definition json file")
@click.argument("vision_name", help="path to vision definition json file")
@click.argument("write_path", help="s3:// or local path to write results to")
@click.argument("read_path", help="s3:// or local path to read predictions from")
@click.option("-l", "--local", is_flag=True, "flag to compute results locally")
@click.option("-d", "--debug", is_flag=True, "flag to increase verbosity of console output")
@vision.command()
def validate(
        vision_definition,
        vision_name,
        keep_instances_alive,
        ec2_keypair_name,
        write_path,
        read_path,
        local,
        debug,
):
    cli_validate_vision(
        s3_fs=s3fs.S3FileSystem(),
        vision_definition=json.load(vision_definition),
        write_path=write_path,
        read_path=read_path,
        vision_name=vision_name,
        ec2_keypair_name=ec2_keypair_name,
        keep_instances_alive=keep_instances_alive,
        local=local,
        debug=debug,
    )
