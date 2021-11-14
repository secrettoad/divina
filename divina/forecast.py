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
        features = fit_model_params["features"]

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

        factor_df = dd.from_array(forecast_df[features].to_dask_array(lengths=True) * da.from_array(fit_model.coef_))
        factor_df.columns = ["factor_{}".format(c) for c in features]
        for c in factor_df:
            forecast_df[c] = factor_df[c]

        if "confidence_intervals" in forecast_definition:
            bootstrap_model_paths = [p for p in read_ls("{}/models/bootstrap".format(
                read_path
            )) if '.' not in p]
            bootstrap_model_paths.sort()

            def load_and_predict_bootstrap_model(paths, link_function, df):
                for path in paths:
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
                        bootstrap_features = bootstrap_params['features']
                    if link_function == 'log':
                        df['{}_h_{}_pred_b_{}'.format(forecast_definition["target"], h,
                                                      path.split("-")[-1])] = da.expm1(
                            bootstrap_model.predict(
                                dd.from_pandas(df[bootstrap_features], chunksize=10000).to_dask_array(
                                    lengths=True)))
                    else:
                        df['{}_h_{}_pred_b_{}'.format(forecast_definition["target"], h,
                                                      path.split("-")[-1])] = bootstrap_model.predict(
                            dd.from_pandas(df[bootstrap_features], chunksize=10000).to_dask_array(lengths=True))

                return df

            if "link_function" in forecast_definition:
                forecast_df = forecast_df.map_partitions(partial(load_and_predict_bootstrap_model,
                                                                 bootstrap_model_paths,
                                                                 forecast_definition['link_function']))

            else:
                forecast_df = forecast_df.map_partitions(partial(load_and_predict_bootstrap_model,
                                                                 bootstrap_model_paths,
                                                                 None))

            df_interval = dd.from_array(dd.from_array(
                forecast_df[['{}_h_{}_pred_b_{}'.format(forecast_definition["target"], h, i.split("-")[-1]) for i in
                             bootstrap_model_paths] + [
                                '{}_h_{}_pred'.format(forecast_definition["target"],
                                                      h)]].to_dask_array(lengths=True).T).repartition(
                npartitions=forecast_df.npartitions).quantile(
                [i * .01 for i in forecast_definition['confidence_intervals']]).to_dask_array(lengths=True).T)
            df_interval.columns = ['{}_h_{}_pred_c_{}'.format(forecast_definition["target"], h, c) for c in
                                   forecast_definition["confidence_intervals"]]

            df_interval = df_interval.repartition(divisions=forecast_df.divisions)
            for c in df_interval.columns:
                forecast_df[c] = df_interval[c]

        forecast_df[forecast_definition["time_index"]] = dd.to_datetime(forecast_df[forecast_definition["time_index"]])

        dd.to_parquet(
            forecast_df,
            "{}/forecast".format(
                write_path,
            )
        )
