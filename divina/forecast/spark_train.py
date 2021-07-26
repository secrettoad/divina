import os
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StringType
import sys
import json
from pyspark.ml.pipeline import Pipeline
from divina.models.preprocessing.base import CategoricalEncoder
from divina.models.ensembles.linear import GLASMA


def train(spark_context, data_definition):

    sqlc = pyspark.sql.SQLContext(spark_context, data_definition)

    df = sqlc.read.format("parquet").load(
        "s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/partitions/endo/*".format(
            os.environ['VISION_ID']))

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
        "sudo aws s3 cp {} {}".format('/home/hadoop/data_definition.json',
                                      's3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/data_definition.json'.format(
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
            df = df.withColumn('{}_h_{}'.format(data_definition['target'], h),
                               F.lag(data_definition['target'], -1 * h, default=None).over(
                                   Window.partitionBy(data_definition['signal_dimensions']).orderBy(
                                       data_definition['time_index'])))
        else:
            df = df.withColumn('{}_h_{}'.format(data_definition['target'], h),
                               F.lag(data_definition['target'], -1 * h, default=None).over(
                                   Window.orderBy(data_definition['time_index'])))

    df = df.withColumn(data_definition['time_index'], F.to_timestamp(data_definition['time_index']))

    categorical_encoder = CategoricalEncoder(discrete_features)

    vector_assembler = VectorAssembler(
        inputCols=['{}_ohe_vec'.format(c.name) for c in discrete_features] + [c.name for c in df.schema if
                                                                              not c.name in non_features + [
                                                                                  c.name for c in discrete_features]],
        outputCol='features')

    sys.stdout.write('Spark model loaded\n')

    for s in data_definition['time_validation_splits']:
        df_train = df.filter(df[data_definition['time_index']] < s)

        for h in data_definition['time_horizons']:
            glasma = GLASMA(feature_col='features', label_col='{}_h_{}'.format(data_definition['target'], h))

            fit_pipeline = Pipeline(stages=[categorical_encoder, vector_assembler, glasma]).fit(df_train)

            sys.stdout.write('Pipeline fit for horizon {}\n'.format(h))

            fit_pipeline.write().overwrite().save(
                "s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/models/s-{}_h-{}".format(
                    os.environ['VISION_ID'], s, h))

            sys.stdout.write('Pipeline persisted for horizon {}\n'.format(h))
