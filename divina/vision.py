from .errors import InvalidDataDefinitionException
import pandas as pd
import json

####TODO abtract rootish from role jsons - use os.path.expandvars
supported_models = ["LinearRegression"]


def get_parameters(s3_fs, model_path):
    with s3_fs.open(
            '{}_params'.format(model_path),
            "rb"
    ) as f:
        params = json.load(f)
        return params


def set_parameters(s3_fs, model_path, params):
    with s3_fs.open(
            '{}_params'.format(model_path),
            "rb"
    ) as f:
        parameters = json.load(f)['params']
    if not params.keys() <= parameters.keys():
        raise Exception('Parameters {} not found in trained model. Cannot set new values for these parameters'.format(
            ', '.join(list(set(params.keys()) - set(parameters.keys())))))
    else:
        parameters.update(params)
        with s3_fs.open(
                '{}_params'.format(model_path),
                "w"
        ) as f:
            json.dump({'params': parameters}, f)
