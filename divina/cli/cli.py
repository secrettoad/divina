import click
from .. import predict, train, validate
from ..dataset import _build
import boto3
from ..train import dask_train
from ..aws.utils import create_modelling_emr
import s3fs
from dask_cloudprovider.aws import EC2Cluster
from dask.distributed import Client
from dask_ml.linear_model import LinearRegression


def cli_build_dataset(dataset_name, write_path, read_path, commit='main', ec2_keypair_name=None, keep_instances_alive=False, local=False):
    session = boto3.session.Session()
    s3_fs = s3fs.S3FileSystem()

    ec2_client = session.client('ec2', 'us-east-2')
    pricing_client = session.client('pricing', region_name='us-east-1')

    _build(commit=commit, dataset_name=dataset_name, write_path=write_path, ec2_client=ec2_client, pricing_client=pricing_client, ec2_keypair_name=ec2_keypair_name,
                                                     keep_instances_alive=keep_instances_alive, read_path=read_path, s3_fs=s3_fs, local=local)


def cli_train_vision(vision_definition, write_path, vision_name, commit='main', ec2_keypair_name=None, keep_instances_alive=False, local=False, dask_address=None):
    s3_fs = s3fs.S3FileSystem()
    dask_model = LinearRegression
    if local:
        with Client() as dask_client:
            dask_train(s3_fs=s3_fs, dask_client=dask_client, dask_model=dask_model, vision_definition=vision_definition,
                       divina_directory=write_path, vision_id=vision_name)
    elif not dask_address:
        with EC2Cluster(key_name=ec2_keypair_name) as cluster:
            with Client(cluster) as dask_client:
                dask_train(s3_fs=s3_fs, dask_client=dask_client, dask_model=dask_model, vision_definition=vision_definition, divina_directory=write_path, vision_id=vision_name)
    else:
        with Client(dask_address) as dask_client:
                dask_train(s3_fs=s3_fs, dask_client=dask_client, dask_model=dask_model, vision_definition=vision_definition, divina_directory=write_path, vision_id=vision_name)

@click.group()
def divina():
    pass

@divina.group()
def dataset():
    pass


@divina.group()
def vision():
    pass

@click.argument('ec2_keypair_name', default=None, required=False)
@click.argument('keep_instances_alive', default=False, required=False)
@click.argument('commit', default='main', required=False)
@click.argument('dataset_name')
@click.argument('write_path')
@click.argument('read_path')
@click.option('-l', '--local', is_flag=True)
@dataset.command()
def build(dataset_name, write_path, read_path, ec2_keypair_name, keep_instances_alive, commit, local):
    if not read_path[:5] == 's3://' and write_path[:5] == 's3://':
        raise Exception('both read_path and write_path must begin with \'s3://\'')
    cli_build_dataset(dataset_name=dataset_name, write_path=write_path, read_path=read_path, ec2_keypair_name=ec2_keypair_name, keep_instances_alive=keep_instances_alive, commit=commit, local=local)



@click.argument('ec2_keypair_name', default=None, required=False)
@click.argument('keep_instances_alive', default=False, required=False)
@click.argument('commit', default='main', required=False)
@click.option('-l', '--local', is_flag=True)
@click.argument('data_definition', type=click.File('rb'))
@click.argument('vision_name')
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
