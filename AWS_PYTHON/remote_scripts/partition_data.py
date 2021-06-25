import pandas as pd
import numpy as np
import sys
import json
import os
from zipfile import ZipFile
from io import BytesIO, StringIO
import re
import s3fs
import traceback
s3_fs = s3fs.S3FileSystem()

sys.stdout.write('RUNNING PYTHON SCRIPT')


class FileTypeNotSupported(Exception):
    """Exception raised when files are supplied by the users of a filetype not supported.

    Attributes:
        extension -- extension of the file not supported
        message -- explanation of the error
    """

    def __init__(self, extension, message="Extension: {} not supported. Please supply either csv or npz files.\n"):
        self.extension = extension
        self.message = message.format(extension)
        super().__init__(self.message)

    def __str__(self):
        return self.message


def partition_data(dfs):
    for df in dfs:
        partition_size = 2000000000
        memory_usage = dfs[df].memory_usage(deep=True).sum()
        if int(memory_usage) <= partition_size:
            groupby = [('mono', dfs[df])]
        elif 'PARTITION_DIMENSIONS' in environment:
            groupby = dfs[df].groupby(environment['PARTITION_DIMENSIONS'])
        else:
            num_chunks = round((int(memory_usage) / partition_size)) + 1
            groupby = zip(range(num_chunks), np.array_split(dfs[df], num_chunks))
        for name, group in groupby:
            sys.stdout.write('name: {} group: {}'.format(name, group))
            save_data_to_s3(name, group)

        ####TODO make this return a list of groups and then from the function above call save_data_to_s3. reference to key is out of scope in
        ####TODO save_data_to_s3


def save_data_to_s3(name, df):
    df.to_parquet('s3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/partitions/{}/partition-{}.parquet'.format(
        environment['VISION_ID'], key.replace('/', '-'), name), index=False)
    sys.stdout.write('SAVED PARQUET -- {} -- {}'.format(environment['VISION_ID'], name))


def decompress_file(key):
    data = s3_fs.open(os.path.join('s3://', environment['SOURCE_BUCKET'], key)).read()
    files = []
    if key.split('.')[-1] == '.zip':
        zip_file = ZipFile(BytesIO(data))
        for name in zip_file.namelist():
            with open('/home/ec2-user/data/{}'.format(os.path.join(key.split('.')[:-1], name)), 'wb+') as f:
                f.write(data)
            files.append('/home/ec2-user/data/{}'.format(os.path.join(key.split('.')[:-1], name)))
    else:
        with open('/home/ec2-user/data/{}'.format(key), 'wb+') as f:
            f.write(data)
        files = ['/home/ec2-user/data/{}'.format(key)]
    return files


def parse_files(files):
    dfs = {}
    for file in files:
        try:
            _file = file
            while _file.split('.')[-1] in ['zip']:
                _file = '.'.join(_file.split('.')[:-1])
            extension = _file.split('.')[-1]
            if extension == 'csv':
                df = pd.read_csv(file)
            elif extension == 'npz':
                npz = np.load(BytesIO(file), allow_pickle=True)
                df = pd.DataFrame.from_records([{item: npz[item] for item in npz.files}])
            else:
                raise FileTypeNotSupported(extension)
            try:
                df[environment['TIME_INDEX']] = pd.to_datetime(df[environment['TIME_INDEX']])
            except KeyError:
                raise Exception('Time index: {} not found in {}'.format(environment['TIME_INDEX'], file))
            dfs[file] = df
        except Exception as e:
            if type(e) == FileTypeNotSupported:
                sys.stdout.write(traceback.print_exc())
                sys.stderr.write(traceback.print_exc())
            else:
                raise e
    return dfs


with open('/home/ec2-user/user-data.json') as f:
    environment = json.load(f)
if not os.path.isdir('/home/ec2-user/data'):
    os.mkdir('/home/ec2-user/data')

for key in environment['SOURCE_KEYS']:
    try:
        files = decompress_file(key)
        dfs = parse_files(files)
        partition_data(dfs)
    except Exception as e:
        sys.stdout.write('Could not partition file: {} error: {}\n'.format(key, traceback.print_exc()))

