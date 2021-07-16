import os

scripts_path = os.path.abspath('.')
os.system("sudo python3 -m pip install Cython")
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StringType
import sys
import json
from pyspark.ml.pipeline import Pipeline
from divina.preprocessing.base import CategoricalEncoder
from divina.models.linear import GLASMA

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


sc = get_spark_context()

sqlc = pyspark.sql.SQLContext(sc)

df = sqlc.read.format("parquet").load(
    "s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/partitions/endo/*".format(os.environ['VISION_ID']))

df = df.drop(*data_definition['drop_features'])

if not 'time_validation_splits' in data_definition:
    split = df.select('*',
                      F.percent_rank().over(Window.partitionBy().orderBy(data_definition['time_index'])).alias(
                          'pct_rank_time')).where('pct_rank_time > 0.5').agg(
        {data_definition['time_index']: 'min'}).collect()[0][0]
    data_definition['time_validation_splits'] = [split]
if not 'time_horizons' in data_definition:
    data_definition['time_horizons'] = [1]

with open('/home/hadoop/data_definition.json', 'w+') as f:
    json.dump(data_definition, f)

os.system(
        "sudo aws s3 cp {} {}".format('/home/hadoop/data_definition.json', 's3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/data_definition.json'.format(
                                 os.environ['VISION_ID'])))

non_features = [data_definition['target'], data_definition['time_index']] + [
            '{}_h_{}'.format(
                data_definition['target'], h) for
            h in
            data_definition['time_horizons']]

discrete_features = [c for c in df.schema if not c.name in non_features and (
                c.dataType is StringType or c.name in data_definition['categorical_features'])]

for h in data_definition['time_horizons']:
    if 'signal_dimension' in data_definition:
        df = df.withColumn('{}_h_{}'.format(data_definition['target'], h), F.lag(data_definition['target'], -1 * h, default=None).over(Window.partitionBy(data_definition['signal_dimensions']).orderBy(data_definition['time_index'])))
    else:
        df = df.withColumn('{}_h_{}'.format(data_definition['target'], h), F.lag(data_definition['target'], -1 * h, default=None).over(Window.orderBy(data_definition['time_index'])))

df = df.withColumn(data_definition['time_index'], F.to_timestamp(data_definition['time_index']))

categorical_encoder = CategoricalEncoder(discrete_features)

vector_assembler = VectorAssembler(
            inputCols=['{}_ohe_vec'.format(c.name) for c in discrete_features] + [c.name for c in df.schema if
                                                                                  not c.name in non_features + [
                                                                                      c.name for c in discrete_features]],
            outputCol='features')

df = Pipeline(stages=[categorical_encoder, vector_assembler]).fit(df).transform(df)

sys.stdout.write('Spark dataframe loaded\n')

for s in data_definition['time_validation_splits']:
    df_train = df.filter(df[data_definition['time_index']] < s)
    df_test = df

    for h in data_definition['time_horizons']:

        glasma = GLASMA(features_col='features', target_col='{}_h_{}'.format(data_definition['target'], h))

        fit_glasma = glasma.fit(df)
        sys.stdout.write('Pipeline fit for horizon {}\n'.format(h))

        df_test = fit_glasma.transform(df_test).withColumnRenamed("prediction",
                                                                  '{}_h_{}_pred'.format(data_definition['target'], h))

        sys.stdout.write('Predictions made for horizon {}\n'.format(h))

    df_test.write.mode('overwrite').option("maxRecordsPerFile", 20000).parquet(
        "s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/predictions/split-{}".format(os.environ[
            'VISION_ID'], s))


