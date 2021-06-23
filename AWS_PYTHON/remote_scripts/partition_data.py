import pandas as pd
import numpy as np
import sys
import json
import os

sys.stdout.write('RUNNING PYTHON SCRIPT')


def partition_data(key):
    df = pd.read_csv(os.path.join('s3://', environment['SOURCE_BUCKET'], key))
    partition_size = 2000000000
    memory_usage = df.memory_usage(deep=True).sum()
    if int(memory_usage) <= partition_size:
        groupby = [('mono', df)]
    elif 'PARTITION_DIMENSIONS' in environment:
        groupby = df.groupby(environment['PARTITION_DIMENSIONS'])
    else:
        num_chunks = round((int(memory_usage) / partition_size)) + 1
        groupby = zip(range(num_chunks), np.array_split(df, num_chunks))
    for name, group in groupby:
        sys.stdout.write('name: {} group: {}'.format(name, group))
        save_data_to_s3(name, group)


def save_data_to_s3(name, df):
    df.to_parquet('s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/partitions/{}/partition-{}.parquet'.format(
        environment['VISION_ID'], key.replace('/', '-'), name), index=False)
    sys.stdout.write('SAVED PARQUET -- {}'.format(environment['VISION_ID']))


with open('/home/ec2-user/user-data.json') as f:
    environment = json.load(f)

for key in environment['SOURCE_KEYS']:
    partition_data(key)
