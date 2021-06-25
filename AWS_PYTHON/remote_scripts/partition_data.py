import pandas as pd
import numpy as np
import sys
import json
import os
from zipfile import ZipFile
from io import BytesIO
import s3fs
import traceback
import shutil

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


def partition_data(files):
    for file in files:
        partition_size = 50000000
        memory_usage = file['df'].memory_usage(deep=True).sum()
        if int(memory_usage) <= partition_size:
            groupby = [('mono', file['df'])]
        elif 'PARTITION_DIMENSIONS' in environment:
            groupby = file['df'].groupby(environment['PARTITION_DIMENSIONS'])
        else:
            num_chunks = round((int(memory_usage) / partition_size)) + 1
            groupby = zip(range(num_chunks), np.array_split(file['df'], num_chunks))
        for name, group in groupby:
            file['df'].to_parquet(
                's3://coysu-divina-prototype-visions/coysu-divina-prototype-{}/partitions/{}/partition-{}.parquet'.format(
                    environment['VISION_ID'], file['source_path'].replace('/', '-'), name), index=False)

        ####TODO make this return a list of groups and then from the function above call save_data_to_s3. reference to key is out of scope in
        ####TODO save_data_to_s3

        sys.stdout.write('SAVED PARQUET -- {} -- {}'.format(environment['VISION_ID'], name))


def decompress_file(key):
    data = s3_fs.open(os.path.join('s3://', environment['SOURCE_BUCKET'], key)).read()
    files = []
    if key.split('.')[-1] == 'zip':
        if not os.path.isdir('/home/ec2-user/data/{}'.format(key.replace('.', '-'))):
            os.mkdir('/home/ec2-user/data/{}'.format(key.replace('.', '-')))
        zip_file = ZipFile(BytesIO(data))
        for name in zip_file.namelist():
            local_path = '/home/ec2-user/data/{}'.format(os.path.join(key.replace('.', '-'), name))
            with open(local_path, 'wb+') as f:
                f.write(zip_file.read(name))
            files.append({'source_path': os.path.join(key, name), 'local_path': local_path,
                          'filename': local_path.split('/')[-1]})
    else:
        local_path = '/home/ec2-user/data/{}'.format(key)
        with open(local_path, 'wb+') as f:
            f.write(data)
        files = [{'source_path': key, 'local_path': '/home/ec2-user/data/{}'.format(key),
                  'filename': local_path.split('/')[-1]}]
    return files


def parse_files(files):
    for file in files:
        try:
            extension = file['local_path'].split('.')[-1]
            if extension == 'csv':
                df = pd.read_csv(file['local_path'])
            elif extension == 'npz':
                npz = np.load(file['local_path'], allow_pickle=True)
                df = pd.DataFrame.from_records([{item: npz[item] for item in npz.files}])
            else:
                raise FileTypeNotSupported(extension)
            if not file['source_path'] in [list(k.keys())[0] for k in environment['EXOGENOUS_MAP']['EXOGENOUS_FILES']]:
                try:
                    df[environment['TIME_INDEX']] = pd.to_datetime(df[environment['TIME_INDEX']])
                except KeyError:
                    raise Exception(
                        'Time index: {} not found in {}'.format(environment['TIME_INDEX'], file['local_path']))
            file['df'] = df
        except Exception as e:
                traceback.print_exc()
                raise e

    return files


def rm(f):
    if os.path.isdir(f): return os.rmdir(f)
    if os.path.isfile(f): return os.unlink(f)
    raise TypeError('must be either file or directory')


with open('/home/ec2-user/user-data.json') as f:
    environment = json.load(f)
if not os.path.isdir('/home/ec2-user/data'):
    os.mkdir('/home/ec2-user/data')
else:
    shutil.rmtree('/home/ec2-user/data')
    os.mkdir('/home/ec2-user/data')

for key in environment['SOURCE_KEYS']:
    try:
        files = decompress_file(key)
        files = parse_files(files)
        partition_data(files)
    except Exception as e:
        sys.stdout.write('Could not partition file: {}\n'.format(key))
        traceback.print_exc()
    finally:
        shutil.rmtree('/home/ec2-user/data')
        os.mkdir('/home/ec2-user/data')
