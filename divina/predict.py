import sys
import dask.dataframe as dd
import pandas as pd
import joblib
from .dataset import get_dataset
import backoff
from botocore.exceptions import ClientError
import os
import dask.bag as db
import dask.dataframe as ddf
from functools import partial


@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def dask_predict(s3_fs, forecast_definition, read_path, write_path):
    forecast_kwargs = {}
    for k in ['forecast_start', 'forecast_end']:
        if k in forecast_definition:
            forecast_kwargs.update({k.split('_')[1]: forecast_definition[k]})
    forecast_df = get_dataset(forecast_definition, pad=True, **forecast_kwargs)

    horizon_ranges = [x for x in forecast_definition["time_horizons"] if type(x) == tuple]
    if len(horizon_ranges) > 0:
        forecast_definition["time_horizons"] = [x for x in forecast_definition["time_horizons"] if type(x) == int]
        for x in horizon_ranges:
            forecast_definition["time_horizons"] = set(forecast_definition["time_horizons"] + list(range(x[0], x[1])))

    if write_path[:5] == "s3://":
        if not s3_fs.exists(write_path):
            s3_fs.mkdir(
                write_path,
                create_parents=True,
                region_name=os.environ["AWS_DEFAULT_REGION"],
                acl="private",
            )

    for s in forecast_definition["time_validation_splits"]:

        validate_kwargs = {}
        if 'validate_start' in forecast_definition:
            if pd.to_datetime(str(s)) < pd.to_datetime(str(forecast_definition["validate_start"])):
                validate_kwargs["start"] = pd.to_datetime(str(s))
            else:
                validate_kwargs["start"] = pd.to_datetime(str(forecast_definition["validate_start"]))
        else:
            validate_kwargs["start"] = pd.to_datetime(str(s))
        if 'validate_end' in forecast_definition:
            if pd.to_datetime(str(s)) > pd.to_datetime(str(forecast_definition["validate_end"])):
                raise Exception(
                    "Bad End: {} | Check Dataset Time Range".format(
                        pd.to_datetime(str(forecast_definition['forecast_start']))))
            else:
                validate_kwargs["end"] = pd.to_datetime(str(s))
        validate_df = get_dataset(forecast_definition, pad=True, **validate_kwargs)

        for h in forecast_definition["time_horizons"]:
            with s3_fs.open(
                    "{}/models/s-{}_h-{}".format(
                        read_path,
                        pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
                        h,
                    ),
                    "rb",
            ) as f:
                fit_model = joblib.load(f)
            if "drop_features" in forecast_definition:
                features = [
                    c
                    for c in validate_df.columns
                    if not c
                           in ["{}_h_{}".format(forecast_definition["target"], h) for h in
                               forecast_definition["time_horizons"]]
                           +
                           ["{}_h_{}_pred".format(forecast_definition["target"], h) for h in
                            forecast_definition["time_horizons"]] + [
                               forecast_definition["time_index"],
                               forecast_definition["target"],
                           ]
                           + forecast_definition["drop_features"]
                ]
            else:
                features = [
                    c
                    for c in validate_df.columns
                    if not c
                           in [
                               "{}_h_{}".format(forecast_definition["target"], h)
                               for h in forecast_definition["time_horizons"]
                           ]
                           +
                           ["{}_h_{}_pred".format(forecast_definition["target"], h) for h in
                            forecast_definition["time_horizons"]] +
                           [
                               forecast_definition["time_index"],
                               forecast_definition["target"],
                           ]
                ]
            validate_df[
                "{}_h_{}_pred".format(forecast_definition["target"], h)
            ] = fit_model.predict(validate_df[features].to_dask_array(lengths=True))
            sys.stdout.write("Validation predictions made for split {}\n".format(s))
            forecast_df[
                "{}_h_{}_pred".format(forecast_definition["target"], h)
            ] = fit_model.predict(forecast_df[features].to_dask_array(lengths=True))

            if "confidence_intervals" in forecast_definition:
                bootstrap_model_paths = [p for p in s3_fs.ls("{}/models/bootstrap".format(
                    read_path
                )) if '.' not in p]

                def load_and_predict_bootstrap_model(bootstrap_df, features, path):
                    with s3_fs.open(
                            path,
                            "rb",
                    ) as f:
                        bootstrap_model = joblib.load(f)
                    bootstrap_df[
                        '{}_h_{}_pred_bootstrap'.format(forecast_definition["target"], h)] = bootstrap_model.predict(
                        bootstrap_df[features].to_dask_array(lengths=True))
                    bootstrap_df['bootstrap_index'] = 1
                    bootstrap_df['bootstrap_index'] = bootstrap_df['bootstrap_index'].cumsum()
                    return bootstrap_df

                bootstrap_dfs = db.from_sequence(bootstrap_model_paths).map(
                    partial(load_and_predict_bootstrap_model, forecast_df, features)).compute()
                bootstrap_df = ddf.concat(bootstrap_dfs)

                for c in forecast_definition["confidence_intervals"]:
                    confidence_forecast = bootstrap_df.groupby(
                            'bootstrap_index')['{}_h_{}_pred_bootstrap'.format(forecast_definition["target"], h)].apply(
                            lambda x: x.quantile(c * .01))
                    confidence_forecast = confidence_forecast.repartition(npartitions=forecast_df.npartitions)
                    confidence_forecast = confidence_forecast.reset_index(drop=True)
                    ###TODO start here = figure out column is assigning weirdly
                    forecast_df[
                        "{}_h_{}_pred_c_{}".format(forecast_definition["target"], h, c)] = confidence_forecast
            sys.stdout.write("Blind predictions made for split {}\n".format(s))
        dd.to_parquet(
            validate_df[
                [forecast_definition["time_index"]]
                + [
                    "{}_h_{}_pred".format(forecast_definition["target"], h)
                    for h in forecast_definition["time_horizons"]
                ]
                ],
            "{}/predictions/s-{}".format(
                write_path,
                pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
            )
        )
        dd.to_parquet(
            forecast_df[
                [forecast_definition["time_index"]]
                + [
                    "{}_h_{}_pred".format(forecast_definition["target"], h)
                    for h in forecast_definition["time_horizons"]
                ] + [
                    "{}_h_{}_pred_c_{}".format(forecast_definition["target"], h, c)
                    for c in forecast_definition["confidence_intervals"] for h in forecast_definition["time_horizons"]
                ]
                ],
            "{}/predictions/s-{}_forecast".format(
                write_path,
                pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
            )
        )
