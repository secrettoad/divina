import click
from .. import predict, train, validate
import pkg_resources
from ..dataset import create_partitioning_ec2
import boto3
from ..dataset import build_dataset_ssh
from ..aws import aws_backoff
import s3fs
from botocore.exceptions import ProfileNotFound
import sys


@click.group()
def divina():
    pass


@divina.command()
def forecast(import_bucket, divina_version=pkg_resources.get_distribution('divina')):
    pass


@divina.group()
def dataset():
    pass

@click.option('--verbose', '-v', is_flag=True, help="Print more output.")
@click.argument('ec2_key_path', default=None, required=False)
@click.argument('keep_instances_alive', default=False, required=False)
@click.argument('branch', default='main', required=False)
@click.argument('region', default='us-east-2', required=False)
@click.argument('dataset_name')
@click.argument('write_path')
@click.argument('read_path')
@dataset.command()
def build(dataset_name, write_path, read_path, ec2_key_path, verbose, keep_instances_alive, branch, region):
    if verbose:
        verbose = 3
    try:
        session = boto3.session.Session(profile_name='divina', region_name=region)
        s3_fs = s3fs.S3FileSystem(profile='divina')
    except ProfileNotFound as e:
        sys.stdout.write('"divina" aws profile not found in ~/.aws/.credentials. check out the instructions here on how to add your credentials: TODO')
        quit()

    ec2_client = session.client('ec2')

    instance, paramiko_key = create_partitioning_ec2(vision_session=session, ec2_keyfile=ec2_key_path,
                                                     keep_instances_alive=keep_instances_alive, data_directory=read_path, s3_fs=s3_fs)
    if not build_dataset_ssh(instance=instance, verbosity=verbose, paramiko_key=paramiko_key, dataset_directory=write_path, dataset_id=dataset_name, branch=branch):
        if not keep_instances_alive:
            aws_backoff.stop_instances(instance_ids=[instance['InstanceId']], ec2_client=ec2_client)
        raise Exception('Dataset build failed. For more information, use verbosity=3')
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
