import pyspark
import sys
import os

scripts_path = os.path.abspath('.')
os.system(
    "pip install -r {}".format(os.path.join(os.path.abspath('.'), 'requirements.txt')))


def get_spark_context(app_name):
    # configure
    conf = pyspark.SparkConf()

    # init & return
    sc = pyspark.SparkContext.getOrCreate(conf=conf)

    # s3a config
    sc._jsc.hadoopConfiguration().set('fs.s3a.endpoint',
                                      's3.eu-central-1.amazonaws.com')
    sc._jsc.hadoopConfiguration().set(
        'fs.s3a.aws.credentials.provider',
        'com.amazonaws.auth.InstanceProfileCredentialsProvider,'
        'com.amazonaws.auth.profile.ProfileCredentialsProvider'
    )

    return pyspark.SQLContext(sparkContext=sc)


sc = get_spark_context('test_vision')

sys.stderr.write('spart_context_created!!!!!')
