import os

scripts_path = os.path.abspath('.')
os.system("sudo python3 -m pip install Cython")
import pyspark
import pyspark.sql.functions as F
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
    metrics = {'time_horizons': {}}
    for h in data_definition['time_horizons']:
        metrics['time_horizons'][h] = {}
        df_h = df.withColumn('resid_h_{}'.format(h),
                                           df['{}_h_{}'.format(data_definition['target'], h)] - df[
                                               '{}_h_{}_pred'.format(data_definition['target'], h)])
        df_split = df_h[df[data_definition['time_index']] > s]
        metrics['time_horizons'][h]['mae'] = df_split.select(F.mean(F.abs('resid_h_{}'.format(h)))).collect()[0][0]
    return metrics


sc = get_spark_context()

sqlc = pyspark.sql.SQLContext(sc)
metrics = {}
for s in data_definition['time_validation_splits']:

    df_pred = sqlc.read.format("parquet").load(
        "s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/predictions/split-{}/*".format(os.environ['VISION_ID'], s))

    if 'signal_dimensions' in data_definition:
        metrics[s] = df_pred.partitionBy(data_definition['signal_dimensions']).rdd.mapPartitions(
            lambda x: get_metrics(data_definition, x)).toDF(
            [c.name for c in df_pred.schema])
    else:
        metrics[s] = get_metrics(data_definition, df_pred)

    if not os.path.exists('/home/hadoop/metrics'.format(s)):
        os.mkdir('/home/hadoop/metrics'.format(s))
    with open('/home/hadoop/metrics/split-{}-metrics.json'.format(s), 'w+') as f:
        json.dump(metrics, f)

    os.system(
        "sudo aws s3 cp {} {}".format('/home/hadoop/metrics/split-{}-metrics.json'.format(s), 's3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/metrics/split-{}-metrics.json'.format(
                                     os.environ['VISION_ID'], s)))
