import os

scripts_path = os.path.abspath('.')
os.system("sudo python3 -m pip install Cython")
import pyspark
import sys
import json
from pyspark.ml.pipeline import PipelineModel

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

sys.stdout.write('Spark model loaded\n')

for s in data_definition['time_validation_splits']:
    df_train = df.filter(df[data_definition['time_index']] < s)
    df_test = df

    for h in data_definition['time_horizons']:

        fit_pipeline = PipelineModel.load(
            "s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/models/s-{}_h-{}".format(os.environ['VISION_ID'], s, h))

        df_test = fit_pipeline.transform(df_test).withColumnRenamed("prediction",
                                                                  '{}_h_{}_pred'.format(data_definition['target'], h))

        sys.stdout.write('Predictions made for horizon {}\n'.format(h))

    df_test.write.mode('overwrite').option("maxRecordsPerFile", 20000).parquet(
        "s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/predictions/split-{}".format(os.environ[
            'VISION_ID'], s))


