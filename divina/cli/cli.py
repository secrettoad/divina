import click
from ..utils import get_parameters, set_parameters, validate_experiment_definition
import sys
import json
import s3fs
from utils import get_dask_client
from ..train import _train
from ..forecast import _forecast
from ..validate import _validate
from ..experiment import _experiment

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
@click.argument("aws_workers", default=0, required=False)
@click.argument("random_state", default=None, required=False)
@divina.command()
@get_dask_client
def experiment(experiment_def, keep_alive, ec2_key, write_path, aws_workers, random_state):
    """:write_path: s3:// or aws path to write results to
    :experiment_def: path to experiment definition JSON file
    :keep_alive: flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging.
    :aws_workers: optionally run experiment on EC2 cluster of specified size
    :ec2_key: aws ec2 keypair name to provide access to dask cluster for debugging.
    """
    experiment_def = json.load(experiment_def)['experiment_definition']
    _experiment(
                    experiment_definition=experiment_def,
                    read_path=write_path,
                    write_path=write_path,
                    ec2_keypair_name=ec2_key,
                    keep_instances_alive=keep_alive,
                    random_state=random_state
    )


@click.argument("ec2_keypair_name", default=None, required=False)
@click.argument("keep_instances_alive", default=False, required=False)
@click.argument("experiment_definition", type=click.File("rb"))
@click.argument("write_path", default="divina-forecast", required=False)
@click.argument("aws_workers", default=0, required=False)
@click.argument("random_state", default=None, required=False)
@divina.command()
def train(
        experiment_def,
        keep_alive,
        ec2_key,
        write_path,
        aws_workers,
        random_state
):
    """:write_path: s3:// or aws path to write results to
    :experiment_def: path to experiment definition JSON file
    :keep_alive: flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging.
    :aws_workers: optionally run experiment on EC2 cluster of specified size
    :ec2_key: aws ec2 keypair name to provide access to dask cluster for debugging.
    """
    experiment_def = json.load(experiment_def)['experiment_definition']
    validate_experiment_definition(experiment_def)
    _train(
        experiment_definition=experiment_def,
        write_path=write_path,
        random_state=random_state
    )


@click.argument("ec2_key", default=None, required=False)
@click.argument("keep_alive", default=False, required=False)
@click.argument("experiment_def", type=click.File("rb"))
@click.argument("write_path", default="divina-forecast", required=False)
@click.argument("read_path", default="divina-forecast", required=False)
@click.argument("aws_workers", default=0, required=False)
@divina.command()
def forecast(
        experiment_def,
        keep_alive,
        ec2_key,
        write_path,
        read_path,
        aws_workers,
):
    """:read_path: s3:// or aws path to read trained model fromn
    :write_path: s3:// or aws path to write results to
    :experiment_def: path to experiment definition JSON file
    :keep_alive: flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging.
    :aws_workers: optionally run experiment on EC2 cluster of specified size
    :ec2_key: aws ec2 keypair name to provide access to dask cluster for debugging.
    """
    experiment_def = json.load(experiment_def)['experiment_definition']
    validate_experiment_definition(experiment_def)
    _forecast(
        experiment_definition=experiment_def,
        write_path=write_path,
        read_path=read_path,
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
@click.argument("aws_workers", default=0, required=False)
@divina.command()
def validate(
        experiment_def,
        keep_alive,
        ec2_key,
        write_path,
        read_path,
        aws_workers,
):
    """:read_path: s3:// or aws path to load models from
    :write_path: s3:// or aws path to write results to
    :experiment_def: path to experiment definition JSON file
    :keep_alive: flag to keep ec2 instances in dask cluster alive after completing computation. use for debugging.
    :aws_workers: optionally run experiment on EC2 cluster of specified size
    :ec2_key: aws ec2 keypair name to provide access to dask cluster for debugging.
    """
    experiment_def = json.load(experiment_def)['experiment_definition']
    validate_experiment_definition(experiment_def)
    _validate(
        experiment_definition=experiment_def,
        write_path=write_path,
        read_path=read_path,
    )


@click.argument("model_path", required=True)
@divina.command()
def get_params(
        model_path
):
    """:model_path: s3:// or aws path to model to get parameters from
    """
    get_parameters(model_path=model_path)
    sys.stdout.write(get_parameters(model_path=model_path))


@click.argument("model_path", required=True)
@divina.command()
def set_params(
        model_path,
        params
):
    """:model_path: s3:// or aws path to model to get parameters from
    :params: dictionary of trained model parameters to update
    """

    sys.stdout.write(set_parameters(model_path=model_path, params=params))