import sys
import dask.dataframe as dd
import joblib
from .dataset import _get_dataset
import backoff
from botocore.exceptions import ClientError
import os
from functools import partial
from .utils import validate_forecast_definition
import json
import dask.array as da


@validate_forecast_definition
@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def _forecast(s3_fs, forecast_definition, read_path, write_path):
    forecast_kwargs = {}
    for k in ['forecast_start', 'forecast_end']:
        if k in forecast_definition:
            forecast_kwargs.update({k.split('_')[1]: forecast_definition[k]})
    forecast_df = _get_dataset(forecast_definition, pad=True, **forecast_kwargs)

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

    if read_path[:5] == "s3://":
        read_open = s3_fs.open
        bootstrap_prefix = None
        read_ls = s3_fs.ls
    else:
        read_open = open
        read_ls = os.listdir
        bootstrap_prefix = os.path.join(read_path, 'models', 'bootstrap')

    for h in forecast_definition["time_horizons"]:
        with read_open(
                "{}/models/h-{}".format(
                    read_path,
                    h,
                ),
                "rb",
        ) as f:
            fit_model = joblib.load(f)
        with read_open(
                "{}/models/h-{}_params.json".format(
                    read_path,
                    h,
                ),
                "r",
        ) as f:
            fit_model_params = json.load(f)
        features = list(fit_model_params["params"].keys())

        if "link_function" in forecast_definition:
            if forecast_definition["link_function"] == 'log':
                forecast_df[
                    "{}_h_{}_pred".format(forecast_definition["target"], h)
                ] = da.expm1(fit_model.predict(forecast_df[features].to_dask_array(lengths=True)))
            sys.stdout.write("Forecasts made for horizon {}\n".format(h))
        else:
            forecast_df[
                "{}_h_{}_pred".format(forecast_definition["target"], h)
            ] = fit_model.predict(forecast_df[features].to_dask_array(lengths=True))
            sys.stdout.write("Forecasts made for horizon {}\n".format(h))

        if "confidence_intervals" in forecast_definition:
            bootstrap_model_paths = [p for p in read_ls("{}/models/bootstrap".format(
                read_path
            )) if '.' not in p]

            def load_and_predict_bootstrap_model(paths, intervals, link_function, df):
                for i, path in enumerate(paths):
                    if bootstrap_prefix:
                        model_path = os.path.join(bootstrap_prefix, path)
                    else:
                        model_path = path
                    with read_open(
                            model_path,
                            "rb",
                    ) as f:
                        bootstrap_model = joblib.load(f)
                    with read_open(
                            "{}_params.json".format(
                                model_path),
                            "r",
                    ) as f:
                        bootstrap_params = json.load(f)
                        bootstrap_features = list(bootstrap_params['params'].keys())
                    if link_function == 'log':
                        df['{}_h_{}_pred_b_{}'.format(forecast_definition["target"], h, i)] = da.expm1(
                            bootstrap_model.predict(
                                dd.from_pandas(df[bootstrap_features], chunksize=10000).to_dask_array(
                                    lengths=True)))
                    else:
                        df['{}_h_{}_pred_b_{}'.format(forecast_definition["target"], h,
                                                      i)] = bootstrap_model.predict(
                            dd.from_pandas(df[bootstrap_features], chunksize=10000).to_dask_array(lengths=True))

                df_agg = df[['{}_h_{}_pred_b_{}'.format(forecast_definition["target"], h, i) for i in
                             range(0, len(paths))] + ['{}_h_{}_pred'.format(forecast_definition["target"], h)]].T
                for i in intervals:
                    if i > 50:
                        interpolation = 'higher'
                    elif i < 50:
                        interpolation = 'lower'
                    else:
                        interpolation = 'linear'
                    df['{}_h_{}_pred_c_{}'.format(forecast_definition["target"], h, i)] = df_agg.quantile(i * .01,
                                                                                                          interpolation=interpolation).T
                return df

            if "link_function" in forecast_definition:
                forecast_df = forecast_df.map_partitions(partial(load_and_predict_bootstrap_model,
                                                                 bootstrap_model_paths,
                                                                 forecast_definition['confidence_intervals'],
                                                                 forecast_definition['link_function']))
            else:
                forecast_df = forecast_df.map_partitions(partial(load_and_predict_bootstrap_model,
                                                                 bootstrap_model_paths,
                                                                 forecast_definition['confidence_intervals'],
                                                                 None))

        forecast_df[forecast_definition["time_index"]] = dd.to_datetime(forecast_df[forecast_definition["time_index"]])

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
            "{}/forecast".format(
                write_path,
            )
        )
