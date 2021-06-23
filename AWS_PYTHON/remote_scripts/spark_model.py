import os
scripts_path = os.path.abspath('.')
os.system(
    "sudo python3 -m pip install -r {}".format(os.path.join('/home/hadoop', 'requirements.txt')))
import pyspark
import sys
import io
import boto3



#TODO use environment variables for the requirements path and app name


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
        'com.amazonaws.auth.InstanceProfileCredentialsProvider',
        'com.amazonaws.auth.profile.ProfileCredentialsProvider'
    )

    return pyspark.SparkContext()


sc = get_spark_context('test_vision')

sys.stderr.write('spart_context_created!!!!!')


boto3_session = boto3.session.Session()

s3 = boto3_session.client('s3')

object_response = s3.get_object(
    Bucket='coysu-divina-prototype-visions',
    Key='coysu-divina-prototype-{}/data/AirPassengers.csv'.format(os.environ['VISION_ID'])
)

sys.stderr.write('csv loaded from s3!!!!!')


s3.put_object(
            Body=io.BytesIO(object_response['Body'].read()),
            Bucket='coysu-divina-prototype-visions',
            Key='coysu-divina-prototype-{}/data/test_write.csv'.format(os.environ['VISION_ID']))
