import os
import pdb
scripts_path = os.path.abspath('.')
os.system("sudo python3 -m pip install Cython")
os.system(
    "sudo python3 -m pip install -r {}".format(os.path.join('/home/hadoop/spark_scripts', 'requirements.txt')))
import pyspark
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import sys
import io

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

df = sqlc.read.format("parquet").load("s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/partitions/endo/*".format(os.environ['VISION_ID']))

sys.stdout.write('Spark dataframe loaded')

vectorAssembler = VectorAssembler(inputCols=[c for c in df.schema.fieldNames() if not c in [os.environ['TARGET'], os.environ['TIME_INDEX']]], outputCol = 'features')
vhouse_df = vectorAssembler.transform(df)
vhouse_df = vhouse_df.select(['features', os.environ['TARGET']])
model = LinearRegression(features_col='features', labelCol = os.environ['TARGET'])
fit_model = model.fit(df)

sys.stdout.write('Spark model fit')

object_response = s3.get_object(
    Bucket='coysu-divina-prototype-visions',
    Key='coysu-divina-prototype-{}/data/AirPassengers.csv'.format(os.environ['VISION_ID'])
)

sys.stderr.write('csv loaded from s3!!!!!')


s3.put_object(
            Body=io.BytesIO(object_response['Body'].read()),
            Bucket='coysu-divina-prototype-visions',
            Key='coysu-divina-prototype-{}/data/test_write.csv'.format(os.environ['VISION_ID']))
