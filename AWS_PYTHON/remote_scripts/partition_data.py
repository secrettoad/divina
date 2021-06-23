import os
import pandas as pd
import numpy as np
import sys
import json

sys.stdout.write('RUNNING PYTHON SCRIPT')


def partition_data():
    df = pd.read_csv('s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/data/{}'.format(
        environment['VISION_ID'], environment['DATA_FILE']))
    partition_size = 2000000000
    if int(environment['SIZE']) <= partition_size:
        groupby = [('mono', df)]
    elif 'PARTITION_DIMENSIONS' in environment:
        groupby = df.groupby(environment['PARTITION_DIMENSIONS'])
    else:
        num_chunks = round((int(environment['SIZE']) / partition_size)) + 1
        groupby = zip(range(num_chunks), np.array_split(df, num_chunks))
    for name, group in groupby:
        sys.stdout.write('name: {} group: {}'.format(name, group))
        save_data_to_s3(name, group)


def save_data_to_s3(name, df):
    df.to_parquet('s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/partitions/{}/partition-{}.parquet'.format(
        environment['VISION_ID'], environment['DATA_FILE'].split('.')[0], name), index=False)
    sys.stdout.write('SAVED PARQUET -- {}'.format(environment['VISION_ID']))


with open('/home/ec2-user/user-data.json') as f:
    environment = json.load(f)
partition_data()
