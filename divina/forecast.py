import json
import os

####TODO abtract rootish from role jsons - use os.path.expandvars
supported_models = ["LinearRegression", "PoissonRegression"]


def get_parameters(s3_fs, model_path):
    if model_path[:5] == "s3://":
        if not s3_fs.exists(model_path):
            s3_fs.mkdir(
                model_path,
                create_parents=True,
                region_name=os.environ["AWS_DEFAULT_REGION"],
                acl="private",
            )
        write_open = s3_fs.open

    else:
        write_open = open
    with write_open(
            '{}_params'.format(model_path),
            "rb"
    ) as f:
        params = json.load(f)
        return params


def set_parameters(s3_fs, model_path, params):
    if model_path[:5] == "s3://":
        if not s3_fs.exists(model_path):
            s3_fs.mkdir(
                model_path,
                create_parents=True,
                region_name=os.environ["AWS_DEFAULT_REGION"],
                acl="private",
            )
        write_open = s3_fs.open

    else:
        write_open = open
    with write_open(
            '{}_params'.format(model_path),
            "rb"
    ) as f:
        parameters = json.load(f)['params']
    if not params.keys() <= parameters.keys():
        raise Exception('Parameters {} not found in trained model. Cannot set new values for these parameters'.format(
            ', '.join(list(set(params.keys()) - set(parameters.keys())))))
    else:
        parameters.update(params)
        with write_open(
                '{}_params'.format(model_path),
                "w"
        ) as f:
            json.dump({'params': parameters}, f)
