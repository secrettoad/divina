import click
from ..forecast import vision, dataset, predict, train, validate
import pkg_resources
import pyspark


def get_spark_context_s3a(s3_endpoint):
    # configure
    conf = pyspark.SparkConf()
    # init & return
    sc = pyspark.SparkContext.getOrCreate(conf=conf)

    # s3a config
    sc._jsc.hadoopConfiguration().set('fs.s3a.endpoint',
                                      s3_endpoint)
    sc._jsc.hadoopConfiguration().set(
        'fs.s3a.aws.credentials.provider',
        'com.amazonaws.auth.InstanceProfileCredentialsProvider',
        'com.amazonaws.auth.profile.ProfileCredentialsProvider'
    )

    return sc



@click.group()
def divina():
    pass


@click.argument('divina_version')
@click.argument('import_bucket')
@divina.command()
def forecast(import_bucket, divina_version=pkg_resources.get_distribution('divina')):
    vision.create_vision(divina_version=divina_version, import_bucket=import_bucket)


@divina.command()
def import_data():
    import_data()


@divina.command()
def build_dataset():
    dataset.build_dataset()

@click.argument('s3_endpoint')
@click.argument('data_definition', type=click.File('rb'))
@divina.command()
def train(s3_endpoint, data_definition):
    sc = get_spark_context_s3a(s3_endpoint)
    train.train(spark_context=sc, data_definition=data_definition)

@click.argument('s3_endpoint')
@click.argument('data_definition', type=click.File('rb'))
@divina.command()
def predict(s3_endpoint, data_definition):
    sc = get_spark_context_s3a(s3_endpoint)
    predict.predict(spark_context=sc, data_definition=data_definition)

@click.argument('s3_endpoint')
@click.argument('data_definition', type=click.File('rb'))
@divina.command()
def validate(s3_endpoint, data_definition):
    sc = get_spark_context_s3a(s3_endpoint)
    validate.validate(spark_context=sc, data_definition=data_definition)






