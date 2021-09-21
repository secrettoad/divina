from .errors import InvalidDataDefinitionException
import pandas as pd
import joblib
import json

####TODO abtract rootish from role jsons - use os.path.expandvars
supported_models = ["LinearRegression"]


def validate_forecast_definition(forecast_definition):
    if not "time_index" in forecast_definition:
        raise InvalidDataDefinitionException(
            "required field time_index not found in data definition"
        )
    if not "target" in forecast_definition:
        raise InvalidDataDefinitionException(
            "required field target not found in data definition"
        )
    if "time_validation_splits" in forecast_definition:
        if not type(forecast_definition["time_validation_splits"]) == list:
            raise InvalidDataDefinitionException(
                "time_validation_splits must be a list of date-like strings"
            )
        elif not all(
                [type(x) == str for x in forecast_definition["time_validation_splits"]]
        ):
            raise InvalidDataDefinitionException(
                "time_validation_splits must be a list of date-like strings"
            )
        elif "forecast_start" in forecast_definition:
            if not all(
                    [pd.to_datetime(x) >= forecast_definition["forecast_start"] for x in
                     forecast_definition["time_validation_splits"]]
            ):
                raise InvalidDataDefinitionException(
                    "time_validation_splits must all be greater than the forecast_start parameter"
                )
    else:
        raise InvalidDataDefinitionException(
            "required key 'time_validation_splits' missing from vision definition."
        )
    if "time_horizons" in forecast_definition:
        if not type(forecast_definition["time_horizons"]) == list:
            raise InvalidDataDefinitionException(
                "time_horizons must be a list of integers"
            )
        elif not all([type(x) == int or type(x) == tuple for x in forecast_definition["time_horizons"]]):
            raise InvalidDataDefinitionException(
                "time_horizons must be a list of integers"
            )
        elif not all(len(x) == 2 for x in forecast_definition['time_horizons'] if type(x) == tuple):
            raise InvalidDataDefinitionException(
                "time_horizons range must be a two-element tuple"
            )
        elif not all(x[0] < x[1] for x in forecast_definition['time_horizons'] if type(x) == tuple):
            raise InvalidDataDefinitionException(
                "first element (beginning) of time_horizons range must smaller than second element (end)"
            )
    else:
        raise InvalidDataDefinitionException(
            "required key 'time_horizons' missing from vision definition."
        )
    if "model" in forecast_definition:
        if not forecast_definition["model"] in supported_models:
            raise InvalidDataDefinitionException(
                "Model '{}' is not supported.".format(forecast_definition["model"])
            )
    if "scenarios" in forecast_definition:
        if not all(['values' in forecast_definition['scenarios'][x] for x in forecast_definition['scenarios']]):
            raise InvalidDataDefinitionException(
                "required key 'values' missing from scenario"
            )
        elif not all(['time_range' in forecast_definition['scenarios'][x] for x in forecast_definition['scenarios']]):
            raise InvalidDataDefinitionException(
                "required key 'time_range' missing from scenario"
            )
        if not forecast_definition["model"] in supported_models:
            raise InvalidDataDefinitionException(
                "Model '{}' is not supported.".format(forecast_definition["model"])
            )


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
