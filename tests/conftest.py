import pyspark
import pytest
import pandas as pd
import numpy as np
import boto3
import moto


@pytest.fixture()
def spark_context():
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


@pytest.fixture()
def test_df_1():
    df = pd.DataFrame([{1, 2, 3},
                  {4, 5, 6},
                  {7, 8, 9},
                  {10, 11, 12}])
    df.columns = ['a', 'b', 'c']
    return df


@pytest.fixture()
def test_df_2():
    df = pd.DataFrame([{1, 2, 3},
                  {4, np.NaN, 6},
                  {7, 8, np.NaN},
                  {np.NaN, 11, 12}])
    df.columns = ['a', 'b', 'c']
    return df


@pytest.fixture()
def test_df_3():
    df = pd.DataFrame([{1, 2, 3},
                  {4, 'a', 6},
                  {7, 8, 'b'},
                  {'c', 11, 12}]).astype('str')
    df.columns = ['a', 'b', 'c']
    return df


@pytest.fixture()
def source_sts():
    with moto.mock_sts():
        yield boto3.client("sts")


@pytest.fixture()
def source_s3():
    with moto.mock_s3():
        yield boto3.client("s3")


@pytest.fixture()
def source_ec2():
    with moto.mock_ec2():
        yield boto3.client("ec2")


@pytest.fixture()
def vision_sts():
    with moto.mock_sts():
        yield boto3.client("sts")


@pytest.fixture()
def vision_s3():
    with moto.mock_s3():
        yield boto3.client("s3")


@pytest.fixture()
def vision_ec2():
    with moto.mock_ec2():
        yield boto3.client("ec2")


@pytest.fixture()
def environment(monkeypatch):
    monkeypatch.setenv('ACCOUNT_NUMBER', '1234586968')
    monkeypatch.setenv('VISION_ID', '49857394875')
