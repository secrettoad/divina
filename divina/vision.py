from .errors import InvalidDataDefinitionException


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
    else:
        raise InvalidDataDefinitionException(
            "required key 'time_validation_splits' missing from vision definition."
        )
    if "time_horizons" in forecast_definition:
        if not type(forecast_definition["time_horizons"]) == list:
            raise InvalidDataDefinitionException(
                "time_horizons must be a list of integers"
            )
        elif not all([type(x) == int for x in forecast_definition["time_horizons"]]):
            raise InvalidDataDefinitionException(
                "time_horizons must be a list of integers"
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
