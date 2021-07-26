import os
import pyspark
import pyspark.sql.functions as F
import json


def get_metrics(data_definition, df, s):
    metrics = {'time_horizons': {}}
    for h in data_definition['time_horizons']:
        metrics['time_horizons'][h] = {}
        df_h = df.withColumn('resid_h_{}'.format(h),
                                           df['{}_h_{}'.format(data_definition['target'], h)] - df[
                                               '{}_h_{}_pred'.format(data_definition['target'], h)])
        df_split = df_h[df[data_definition['time_index']] > s]
        metrics['time_horizons'][h]['mae'] = df_split.select(F.mean(F.abs('resid_h_{}'.format(h)))).collect()[0][0]
    return metrics


def validate(spark_context, data_definition):

    sqlc = pyspark.sql.SQLContext(spark_context)
    metrics = {}
    for s in data_definition['time_validation_splits']:

        df_pred = sqlc.read.format("parquet").load(
            "s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/predictions/split-{}/*".format(os.environ['VISION_ID'], s))

        if 'signal_dimensions' in data_definition:
            metrics[s] = df_pred.partitionBy(data_definition['signal_dimensions']).rdd.mapPartitions(
                lambda x: get_metrics(data_definition, x, s)).toDF(
                [c.name for c in df_pred.schema])
        else:
            metrics[s] = get_metrics(data_definition, df_pred, s)

        if not os.path.exists('/home/hadoop/metrics'.format(s)):
            os.mkdir('/home/hadoop/metrics'.format(s))
        with open('/home/hadoop/metrics/split-{}-metrics.json'.format(s), 'w+') as f:
            json.dump(metrics, f)

        os.system(
            "sudo aws s3 cp {} {}".format('/home/hadoop/metrics/split-{}-metrics.json'.format(s), 's3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/metrics/split-{}-metrics.json'.format(
                                         os.environ['VISION_ID'], s)))
