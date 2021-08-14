import click
from .. import predict, train, validate
import pkg_resources
from ..dataset import _build
import boto3
from ..aws import aws_backoff
import s3fs


def cli_build(dataset_name, write_path, read_path, commit, ec2_keypair_name=None, keep_instances_alive=False):
    session = boto3.session.Session()
    s3_fs = s3fs.S3FileSystem()

    ec2_client = session.client('ec2', 'us-east-2')
    pricing_client = session.client('pricing', region_name='us-east-1')

    _build(commit=commit, dataset_name=dataset_name, write_path=write_path, ec2_client=ec2_client, pricing_client=pricing_client, ec2_keypair_name=ec2_keypair_name,
                                                     keep_instances_alive=keep_instances_alive, read_path=read_path, s3_fs=s3_fs)


@click.group()
def divina():
    pass


@divina.command()
def forecast(import_bucket, divina_version=pkg_resources.get_distribution('divina')):
    pass


@divina.group()
def dataset():
    pass


@click.argument('ec2_keypair_name', default=None, required=False)
@click.argument('keep_instances_alive', default=False, required=False)
@click.argument('commit', default='main', required=False)
@click.argument('dataset_name')
@click.argument('write_path')
@click.argument('read_path')
@dataset.command()
def build(dataset_name, write_path, read_path, ec2_keypair_name, keep_instances_alive, commit):
    if not read_path[:5] == 's3://' and write_path[:5] == 's3://':
        raise Exception('both read_path and write_path must begin with \'s3://\'')
    cli_build(dataset_name=dataset_name, write_path=write_path, read_path=read_path, ec2_keypair_name=ec2_keypair_name, keep_instances_alive=keep_instances_alive, commit=commit)


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
