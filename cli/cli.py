import click
from ..forecast import vision, dataset, predict, train, validate
import pkg_resources
from pyspark import SparkContext
import pyspark
from ..forecast.dataset import create_partitioning_ec2
import boto3


def get_spark_context_s3(s3_endpoint):
    # configure
    conf = pyspark.SparkConf()

    sc = SparkContext.getOrCreate(
        conf=conf)

    # s3a config
    sc._jsc.hadoopConfiguration().set('fs.s3.endpoint',
                                      s3_endpoint)
    sc._jsc.hadoopConfiguration().set(
        'fs.s3.aws.credentials.provider',
        'com.amazonaws.auth.InstanceProfileCredentialsProvider',
        'com.amazonaws.auth.profile.ProfileCredentialsProvider'
    )

    return sc


@click.group()
def divina():
    pass


@divina.command()
def forecast(import_bucket, divina_version=pkg_resources.get_distribution('divina')):
    vision.create_vision(divina_version=divina_version, import_bucket=import_bucket)


@divina.command()
def dataset():
    pass

@click.argument('data_path', default='s3://divina-dataset')
@click.argument('ec2_key', default=None)
@click.argument('verbosity', default=0)
@click.argument('keep_instances_alive', default=False)
@dataset.command()
def build(data_path, ec2_keyfile, verbosity, keep_instances_alive):
    session = boto3.Session(profile_name='divina')

    ec2_client = session.client('ec2')

    instance, paramiko_key = create_partitioning_ec2(vision_session=session, ec2_keyfile=ec2_keyfile,
                                                     keep_instances_alive=keep_instances_alive, data_directory=)
    if not build_dataset_ssh(instance=instance, verbosity=verbosity, paramiko_key=paramiko_key,
                             divina_pip_arguments=divina_pip_arguments):
        if not keep_instances_alive:
            aws_backoff.stop_instances(instance_ids=[instance['InstanceId']], ec2_client=vision_ec2_client)
        quit()
    if not keep_instances_alive:
        aws_backoff.stop_instances(instance_ids=[instance['InstanceId']], ec2_client=vision_ec2_client)




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