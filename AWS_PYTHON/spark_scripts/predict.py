import os

scripts_path = os.path.abspath('.')
os.system("sudo python3 -m pip install Cython")
os.system(
    "sudo python3 -m pip install -r {}".format(os.path.join('/home/hadoop/spark_scripts', 'requirements.txt')))
import pyspark
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.sql.types import StringType, DecimalType
from pyspark.ml import Pipeline
import sys
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


sc = get_spark_context()

sqlc = pyspark.sql.SQLContext(sc)

df = sqlc.read.format("parquet").load(
    "s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/partitions/endo/*".format(os.environ['VISION_ID']))

df = df.drop(*data_definition['drop_features'])

sys.stdout.write('Spark dataframe loaded')

non_features = [data_definition['target']]
if 'time_index' in data_definition:
    non_features = non_features + [data_definition['time_index']]

discrete_features = [c for c in df.schema if not c in non_features and (
            c.dataType is StringType or c.name in data_definition['categorical_features'])]
string_indexers = [StringIndexer(inputCol=c.name, outputCol="{}_index".format(c.name)) for c in discrete_features]
one_hot_encoders = [
    OneHotEncoder(dropLast=False, inputCol="{}_index".format(c.name), outputCol="{}_ohe_vec".format(c.name)) for c in
    discrete_features]
vector_assembler = VectorAssembler(
    inputCols=['{}_ohe_vec'.format(c.name) for c in discrete_features] + [c.name for c in df.schema if
                                                                          not c.name in non_features and c.name not in [
                                                                              c.name for c in discrete_features]],
    outputCol='features')
linear_regression = LinearRegression(featuresCol='features', labelCol=data_definition['target'])
model = Pipeline(stages=string_indexers + one_hot_encoders + [vector_assembler, linear_regression])

fit_model = model.fit(df)

sys.stdout.write('Pipeline fit')

df_pred = fit_model.transform(df)

sys.stdout.write('Predictions made')

df_pred.write.option("maxRecordsPerFile", 20000).parquet("s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/predictions".format(os.environ['VISION_ID']))