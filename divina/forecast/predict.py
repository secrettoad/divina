import os
import pyspark
import sys
from pyspark.ml.pipeline import PipelineModel


def predict(data_definition, spark_context):

    sqlc = pyspark.sql.SQLContext(spark_context)

    df = sqlc.read.format("parquet").load(
        "s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/partitions/endo/*".format(
            os.environ['VISION_ID']))

    df = df.drop(*data_definition['drop_features'])

    sys.stdout.write('Spark model loaded\n')

    for s in data_definition['time_validation_splits']:

        for h in data_definition['time_horizons']:
            fit_pipeline = PipelineModel.load(
                "s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/models/s-{}_h-{}".format(
                    os.environ['VISION_ID'], s, h))

            df = fit_pipeline.transform(df).withColumnRenamed("prediction",
                                                              '{}_h_{}_pred'.format(data_definition['target'], h))

            sys.stdout.write('Predictions made for horizon {}\n'.format(h))

        df.write.mode('overwrite').option("maxRecordsPerFile", 20000).parquet(
            "s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/predictions/split-{}".format(os.environ[
                                                                                                            'VISION_ID'],
                                                                                                        s))


