import os

scripts_path = os.path.abspath('.')
os.system("sudo python3 -m pip install Cython")
os.system(
    "sudo python3 -m pip install -r {}".format(os.path.join('/home/hadoop/spark_scripts', 'requirements.txt')))
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import Window
import json

with open('/home/hadoop/data_definition.json') as f:
    data_definition = json.load(f)


def get_spark_context():
    # configure
    conf = pyspark.SparkConf()
    # init & return
    sc = pyspark.SparkContext.getOrCreate(conf=conf)

    # s3a config
    sc._jsc.hadoopConfiguration().set('fs.s3a.endpoint',
                                      's3.us-east-2.amazonaws.com')
    sc._jsc.hadoopConfiguration().set(
        'fs.s3a.aws.credentials.provider',
        'com.amazonaws.auth.InstanceProfileCredentialsProvider',
        'com.amazonaws.auth.profile.ProfileCredentialsProvider'
    )

    return sc

    ###TODO TODO REPLACE FOR LOOPS WITH PROPER PARALLELIZATION


def get_metrics(data_definition, df):
    metrics = {}
    for h in data_definition['time_horizons']:
        df_h = df.withColumn('resid_h_{}'.format(h),
                                           df['{}_h_{}'.format(data_definition['target'], h)] - df[
                                               '{}_h_{}_pred'.format(data_definition['target'], h)])
        for s in data_definition['time_validation_splits']:
            df_split = df_h[df[data_definition['time_index']] > s]
            metrics[s] = {'time_horizons': {h:{}}}
            metrics[s]['time_horizons'][h]['mae'] = F.mean(F.abs(df_split['resid_h_{}'.format(h)]))
    return metrics


if not 'train_only' in data_definition or not data_definition['train_only']:
    sc = get_spark_context()

    sqlc = pyspark.sql.SQLContext(sc)

    df_pred = sqlc.read.format("parquet").load(
        "s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/predictions/*".format(os.environ['VISION_ID']))

    if not 'time_validation_splits' in data_definition:
        split = df_pred.select('*', F.percent_rank().over(Window.partitionBy().orderBy(data_definition['time_index'])).alias('pct_rank_time')).where('pct_rank_time > 0.7').agg({data_definition['time_index']: 'min'}).collect()[0][0]
        data_definition['time_validation_splits'] = [split]
    if not 'time_horizons' in data_definition:
        data_definition['time_horizons'] = [1]

    if 'signal_dimensions' in data_definition:
        metrics = df_pred.partitionBy(data_definition['signal_dimensions']).rdd.mapPartitions(
            lambda x: get_metrics(data_definition, x)).toDF(
            [c.name for c in df_pred.schema])
    else:
        metrics = get_metrics(data_definition, df_pred)

    with open('/home/hadoop/metrics.json', 'wb+') as f:
        json.dump(metrics, f)

    os.system(
        "sudo aws s3 cp {} {}".format('/home/hadoop/data_definition.json', 's3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/metrics.json'.format(
                                 os.environ['VISION_ID'])))
