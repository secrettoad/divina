import click
from ..train import _train
from ..forecast import _forecast
from ..utils import get_parameters, set_parameters
from ..validate import _validate
from dask_cloudprovider.aws import EC2Cluster
from dask.distributed import Client
from botocore.exceptions import NoRegionError
from botocore.exceptions import NoCredentialsError
import os
import sys
import json
import s3fs


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


def cli_train_experiment(
        s3_fs,
        experiment_definition,
        write_path,
        ec2_keypair_name=None,
        keep_instances_alive=False,
        aws=False,
        debug=False,
        dask_client=None,
        random_state=None
):
    if not aws:
        with Client():
            _train(
                s3_fs=s3_fs,
                experiment_definition=experiment_definition,
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
                    cluster.scale(10)
                    with Client(cluster):
                        _train(
                            s3_fs=s3_fs,
                            experiment_definition=experiment_definition,
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
            cluster.scale(10)
            with Client(cluster):
                _train(
                    s3_fs=s3_fs,
                    experiment_definition=experiment_definition,
                    write_path=write_path,
                    random_seed=random_state
                )

    else:
        _train(
            s3_fs=s3_fs,
            experiment_definition=experiment_definition,
            write_path=write_path,
            random_seed=random_state
        )


def cli_forecast_experiment(
        s3_fs,
        experiment_definition,
        write_path,
        read_path,
        ec2_keypair_name=None,
        keep_instances_alive=False,
        aws=False,
        debug=False,
        dask_client=None
):
    if not aws:
        with Client():
            _forecast(
                s3_fs=s3_fs,
                experiment_definition=experiment_definition,
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
                    cluster.scale(10)
                    with Client(cluster):
                        _forecast(
                            s3_fs=s3_fs,
                            experiment_definition=experiment_definition,
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
            cluster.scale(10)
            with Client(cluster):
                _forecast(
                    s3_fs=s3_fs,
                    experiment_definition=experiment_definition,
                    write_path=write_path,
                    read_path=read_path
                )

    else:
        _forecast(
            s3_fs=s3_fs,
            experiment_definition=experiment_definition,
            write_path=write_path,
            read_path=read_path
        )


def cli_validate_experiment(
        s3_fs,
        experiment_definition,
        write_path,
        read_path,
        ec2_keypair_name=None,
        keep_instances_alive=False,
        aws=False,
        debug=False,
        dask_client=None,
):
    if not aws:
        with Client():
            _validate(
                s3_fs=s3_fs,
                experiment_definition=experiment_definition,
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
                    cluster.scale(50)
                    with Client(cluster):
                        _validate(
                            s3_fs=s3_fs,
                            experiment_definition=experiment_definition,
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
            cluster.scale(50)
            with Client(cluster):
                _validate(
                    s3_fs=s3_fs,
                    experiment_definition=experiment_definition,
                    write_path=write_path,
                    read_path=read_path,
                )

    else:
        _validate(
            s3_fs=s3_fs,
            experiment_definition=experiment_definition,
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
@click.argument("experiment_def", type=click.File("rb"))
@click.option("--aws", is_flag=True, help="flag to compute results awsly")
@click.option(
    "-d", "--debug", is_flag=True, help="flag to increase verbosity of console output"
)
@divina.command()
def experiment(experiment_def, keep_alive, ec2_key, write_path, aws, debug):
    """:write_path: s3:// or aws path to write results to
    :experiment_def: path to experiment definition JSON file
    :keep_alive: flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging.
    :--aws: optionally run experiment on EC2 cluster
    :ec2_key: aws ec2 keypair name to provide access to dask cluster for debugging.
    """
    experiment_def = json.load(experiment_def)['experiment_definition']
    ###TODO create get_dask_client function that accepts ec2key and keep alive
    if aws:
        with EC2Cluster(
                key_name=ec2_key,
                security=False,
                docker_image="jhurdle/divina:latest",
                debug=debug,
                env_vars={
                    "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                    "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                },
                auto_shutdown=not keep_alive,
        ) as cluster:
            cluster.scale(10)
            with Client(cluster) as client:
                cli_train_experiment(
                    s3_fs=s3fs.S3FileSystem(),
                    experiment_definition=experiment_def,
                    write_path=write_path,
                    ec2_keypair_name=ec2_key,
                    keep_instances_alive=keep_alive,
                    dask_client=client,
                    debug=debug,
                )
                cli_forecast_experiment(
                    s3_fs=s3fs.S3FileSystem(),
                    experiment_definition=experiment_def,
                    write_path=write_path,
                    read_path=write_path,
                    ec2_keypair_name=ec2_key,
                    keep_instances_alive=keep_alive,
                    dask_client=client,
                    debug=debug,
                )
                cli_validate_experiment(
                    s3_fs=s3fs.S3FileSystem(),
                    experiment_definition=experiment_def,
                    write_path=write_path,
                    read_path=write_path,
                    ec2_keypair_name=ec2_key,
                    keep_instances_alive=keep_alive,
                    dask_client=client,
                    debug=debug,
                )
    else:
        cli_train_experiment(
            s3_fs=s3fs.S3FileSystem(),
            experiment_definition=experiment_def,
            write_path=write_path,
            ec2_keypair_name=ec2_key,
            keep_instances_alive=keep_alive,
            aws=aws,
            debug=debug,
        )
        cli_forecast_experiment(
            s3_fs=s3fs.S3FileSystem(),
            experiment_definition=experiment_def,
            write_path=write_path,
            read_path=write_path,
            ec2_keypair_name=ec2_key,
            keep_instances_alive=keep_alive,
            aws=aws,
            debug=debug,
        )
        cli_validate_experiment(
            s3_fs=s3fs.S3FileSystem(),
            experiment_definition=experiment_def,
            write_path=write_path,
            read_path=write_path,
            ec2_keypair_name=ec2_key,
            keep_instances_alive=keep_alive,
            aws=aws,
            debug=debug,
        )


@click.argument("ec2_keypair_name", default=None, required=False)
@click.argument("keep_instances_alive", default=False, required=False)
@click.argument("experiment_definition", type=click.File("rb"))
@click.argument("write_path", default="divina-forecast", required=False)
@click.option("-l", "--aws", is_flag=True, help="flag to compute results awsly")
@click.option(
    "-d", "--debug", is_flag=True, help="flag to increase verbosity of console output"
)
@divina.command()
def train(
        experiment_def,
        keep_alive,
        ec2_key,
        write_path,
        aws,
        debug,
):
    """:write_path: s3:// or aws path to write results to
    :experiment_def: path to experiment definition JSON file
    :keep_alive: flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging.
    :--aws: optionally run experiment on EC2 cluster
    :ec2_key: aws ec2 keypair name to provide access to dask cluster for debugging.
    """
    cli_train_experiment(
        s3_fs=s3fs.S3FileSystem(),
        experiment_definition=json.load(experiment_def),
        write_path=write_path,
        ec2_keypair_name=ec2_key,
        keep_instances_alive=keep_alive,
        aws=aws,
        debug=debug,
    )


@click.argument("ec2_key", default=None, required=False)
@click.argument("keep_alive", default=False, required=False)
@click.argument("experiment_def", type=click.File("rb"))
@click.argument("write_path", default="divina-forecast", required=False)
@click.argument("read_path", default="divina-forecast", required=False)
@click.option("-l", "--aws", is_flag=True, help="flag to compute results awsly")
@click.option(
    "-d", "--debug", is_flag=True, help="flag to increase verbosity of console output"
)
@divina.command()
def forecast(
        experiment_def,
        keep_alive,
        ec2_key,
        write_path,
        read_path,
        aws,
        debug,
):
    """:read_path: s3:// or aws path to read trained model fromn
    :write_path: s3:// or aws path to write results to
    :experiment_def: path to experiment definition JSON file
    :keep_alive: flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging.
    :--aws: optionally run experiment on EC2 cluster
    :ec2_key: aws ec2 keypair name to provide access to dask cluster for debugging.
    """
    cli_forecast_experiment(
        s3_fs=s3fs.S3FileSystem(),
        experiment_definition=json.load(experiment_def),
        write_path=write_path,
        read_path=read_path,
        ec2_keypair_name=ec2_key,
        keep_instances_alive=keep_alive,
        aws=aws,
        debug=debug,
    )


@click.argument(
    "ec2_key",
    default=None,
    required=False,
)
@click.argument("keep_alive", default=False, required=False)
@click.argument("experiment_def", type=click.File("rb"))
@click.argument("write_path", default="divina-forecast", required=False)
@click.argument("read_path", default="divina-forecast", required=False)
@click.option("-l", "--aws", is_flag=True, help="flag to compute results awsly")
@click.option(
    "-d", "--debug", is_flag=True, help="flag to increase verbosity of console output"
)
@divina.command()
def validate(
        experiment_def,
        keep_alive,
        ec2_key,
        write_path,
        read_path,
        aws,
        debug,
):
    """:read_path: s3:// or aws path to load models from
    :write_path: s3:// or aws path to write results to
    :experiment_def: path to experiment definition JSON file
    :keep_alive: flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging.
    :--aws: optionally run experiment on EC2 cluster
    :ec2_key: aws ec2 keypair name to provide access to dask cluster for debugging.
    """
    cli_validate_experiment(
        s3_fs=s3fs.S3FileSystem(),
        experiment_definition=json.load(experiment_def),
        write_path=write_path,
        read_path=read_path,
        ec2_keypair_name=ec2_key,
        keep_instances_alive=keep_alive,
        aws=aws,
        debug=debug,
    )


@click.argument("model_path", required=True)
@divina.command()
def get_params(
        model_path
):
    """:model_path: s3:// or aws path to model to get parameters from
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
    """:model_path: s3:// or aws path to model to get parameters from
    :params: dictionary of trained model parameters to update
    """
    cli_set_params(
        s3_fs=s3fs.S3FileSystem(),
        model_path=model_path,
        params=params
    )
