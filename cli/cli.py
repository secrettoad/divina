import click
from ..forecast import vision, predict, train, validate
import pkg_resources
from ..forecast.dataset import create_partitioning_ec2
import boto3
from ..forecast.dataset import build_dataset_ssh
from ..aws import aws_backoff
import s3fs


@click.group()
def divina():
    pass


@divina.command()
def forecast(import_bucket, divina_version=pkg_resources.get_distribution('divina')):
    vision.create_vision(divina_version=divina_version, import_bucket=import_bucket)


@divina.command()
def dataset():
    pass


@click.argument('dataset_name')
@click.argument('write_path')
@click.argument('read_path')
@click.argument('ec2_key_path', default=None)
@click.argument('verbosity', default=0)
@click.argument('keep_instances_alive', default=False)
@click.argument('branch', default='main')
@dataset.command()
def build(dataset_name, write_path, read_path, ec2_key_path, verbosity, keep_instances_alive, branch):

    session = boto3.Session(profile_name='divina')

    s3_fs = s3fs.S3FileSystem(profile='divina')

    ec2_client = session.client('ec2')

    instance, paramiko_key = create_partitioning_ec2(vision_session=session, ec2_keyfile=ec2_key_path,
                                                     keep_instances_alive=keep_instances_alive, data_directory=read_path, s3_fs=s3_fs)
    if not build_dataset_ssh(instance=instance, verbosity=verbosity, paramiko_key=paramiko_key, dataset_directory=write_path, dataset_id=dataset_name, branch=branch):
        if not keep_instances_alive:
            aws_backoff.stop_instances(instance_ids=[instance['InstanceId']], ec2_client=ec2_client)
        quit()
    if not keep_instances_alive:
        aws_backoff.stop_instances(instance_ids=[instance['InstanceId']], ec2_client=ec2_client)


@click.argument('s3_endpoint')
@click.argument('data_definition', type=click.File('rb'))
@click.argument('vision_id', envvar='VISION_ID')
@divina.command()
def train(s3_endpoint, data_definition, vision_id):
    sc = get_spark_context_s3(s3_endpoint)
    train.train(spark_context=sc, data_definition=data_definition, vision_id=vision_id)


@click.argument('s3_endpoint')
@click.argument('data_definition', type=click.File('rb'))
@divina.command()
def predict(s3_endpoint, data_definition):
    sc = get_spark_context_s3(s3_endpoint)
    predict.predict(spark_context=sc, data_definition=data_definition)


@click.argument('s3_endpoint')
@click.argument('data_definition', type=click.File('rb'))
@divina.command()
def validate(s3_endpoint, data_definition):
    sc = get_spark_context_s3(s3_endpoint)
    validate.validate(spark_context=sc, data_definition=data_definition)
