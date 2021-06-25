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
    if key.split('.')[-1] == '.zip':
        zip_file = ZipFile(BytesIO(data))
        files = {os.path.join(key.split('.')[:-1], name): zip_file.read(name) for name in zip_file.namelist()}
    else:
        b = data
        files = {key: b}
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
                failed_encodings = []
                ###current scope is that all files supplied are encoded as below
                encodings = ['utf_8', 'ascii', 'latin_1']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(StringIO(files[file].decode(encoding, errors='replace')), encoding=encoding)
                        break
                    except Exception:
                        failed_encodings += [encoding]
                        sys.stdout.write('{} is not encoded via {}. Trying next encoding...\n'.format(file, encoding))
                if set(failed_encodings) == set(encodings):
                    message = 'Could not decode {}. Please encode the file in utf-8, ascii or latin-1'.format(file)
                    raise Exception(message)
            elif extension == 'npz':
                npz = np.load(BytesIO(files[file]), allow_pickle=True, encoding=encoding)
                df = pd.DataFrame.from_records([{item: npz[item] for item in npz.files}])
            else:
                raise FileTypeNotSupported(extension)
            if not any(re.match(map_file.replace('*', '/^\w+$/'), file) for map_file in [[k for k in e][0] for e in environment['EXOGENOUS_MAP']['EXOGENOUS_FILES']]):
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

for key in environment['SOURCE_KEYS']:
    try:
        files = decompress_file(key)
        dfs = parse_files(files)
        partition_data(dfs)
    except Exception as e:
        sys.stdout.write('Could not partition file: {} error: {}\n'.format(key, traceback.print_exc()))

