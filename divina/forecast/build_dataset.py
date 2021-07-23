import os
import sys
import pandas as pd
import numpy as np
import json
from zipfile import ZipFile
from io import BytesIO
import s3fs
import traceback
import shutil
s3_fs = s3fs.S3FileSystem(region_name='us-east-2')

sys.stdout.write('RUNNING PYTHON SCRIPT\n')


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


def partition_data(data_definition, files):
    for file in files:
        partition_size = 5000000000
        memory_usage = file['df'].memory_usage(deep=True).sum()
        if file['source_path'] in [list(k.keys())[0] for k in data_definition['exogenous_files']]:
            data_type = 'exog'
        else:
            data_type = 'endo'
        root_path = 's3://{}/coysu-divina-prototype-{}/partitions/{}'.format(os.environ['DIVINA_BUCKET'],
                    os.environ['VISION_ID'], data_type)
        if int(memory_usage) <= partition_size:
            path = root_path + '/{}.parquet'.format(file['source_path'])
            file['df'].to_parquet(path, index=False)
        elif 'signal_dimensions' in data_definition:
            partition_cols = data_definition['signal_dimensions']
            file['df'].to_parquet(
                root_path, index=False, partition_cols=partition_cols)
        else:
            partition_rows = partition_size/memory_usage * len(file['df'])
            file['df']['partition'] = np.arange(len(file['df'])) // partition_rows
            partition_cols = ['partition']
            file['df'].to_parquet(
                root_path, index=False, partition_cols=partition_cols)
        sys.stdout.write('SAVED PARQUET - {} - {}\n'.format(os.environ['VISION_ID'], file['source_path']))


def decompress_file(tmp_dir, key):
    data = s3_fs.open(os.path.join('s3://', os.environ['DIVINA_BUCKET'], key)).read()
    files = []
    if key.split('.')[-1] == 'zip':
        if not os.path.isdir('{}/{}'.format(tmp_dir, key.split('/')[-1].replace('.', '-'))):
            os.mkdir('{}/{}'.format(tmp_dir, key.split('/')[-1].replace('.', '-')))
        zip_file = ZipFile(BytesIO(data))
        for name in zip_file.namelist():
            local_path = '{}/{}'.format(tmp_dir, os.path.join(key.split('/')[-1].replace('.', '-'), name))
            with open(local_path, 'wb+') as f:
                f.write(zip_file.read(name))
            files.append({'source_path': os.path.join(key, name), 'local_path': local_path,
                          'filename': local_path.split('/')[-1]})
    else:
        local_path = '{}/{}'.format(tmp_dir, key.split('/')[-1])
        with open(local_path, 'wb+') as f:
            f.write(data)
        files = [{'source_path': key, 'local_path': local_path,
                  'filename': local_path.split('/')[-1]}]
    return files


def parse_files(files):
    for file in files:
        try:
            extension = file['local_path'].split('.')[-1]
            if extension == 'csv':
                df = pd.read_csv(file['local_path'])
            else:
                raise FileTypeNotSupported(extension)
            file['df'] = df
        except Exception as e:
                traceback.print_exc()
                raise e

    return files


def rm(f):
    if os.path.isdir(f): return os.rmdir(f)
    if os.path.isfile(f): return os.unlink(f)
    raise TypeError('must be either file or directory')


def build_dataset(tmp_dir='/tmp/data'):
    if os.path.exists('./user-data.json'):
        with open('./user-data.json') as f:
            os.environ.update(json.load(f)['ENVIRONMENT'])
    with open('../config/data_definition.json') as f:
        data_definition = json.load(f)
    data_directories = ['endo', 'exog']
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    for d in data_directories:
        if not os.path.isdir(os.path.join(tmp_dir, d)):
            os.mkdir(os.path.join(tmp_dir, d))

    for key in s3_fs.ls('s3://{}/coysu-divina-prototype-{}/data'.format(os.environ['DIVINA_BUCKET'], os.environ['VISION_ID'])):
        try:
            files = decompress_file(tmp_dir, key)
            files = parse_files(files)
            partition_data(data_definition, files)
        except Exception as e:
            sys.stdout.write('Could not partition file: {}\n'.format(key))
            traceback.print_exc()
            raise e
        finally:
            shutil.rmtree('{}'.format(tmp_dir))
            for d in data_directories:
                os.mkdir(os.path.join('./', d))
