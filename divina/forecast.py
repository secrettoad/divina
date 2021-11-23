import sys
import dask.dataframe as dd
import joblib
from .dataset import _get_dataset
import backoff
from botocore.exceptions import ClientError
import os
from functools import partial
from .utils import validate_experiment_definition
import json
import dask.array as da
from pandas.api.types import is_numeric_dtype


@validate_experiment_definition
@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def _forecast(s3_fs, experiment_definition, read_path, write_path):
    if not "time_horizons" in experiment_definition:
        time_horizons = [0]
    else:
        time_horizons = experiment_definition["time_horizons"]
    forecast_kwargs = {}
    for k in ['forecast_start', 'forecast_end']:
        if k in experiment_definition:
            forecast_kwargs.update({k.split('_')[1]: experiment_definition[k]})
    forecast_df = _get_dataset(experiment_definition, pad=True, **forecast_kwargs)

    horizon_ranges = [x for x in time_horizons if type(x) == tuple]
    if len(horizon_ranges) > 0:
        time_horizons = [x for x in time_horizons if type(x) == int]
        for x in horizon_ranges:
            time_horizons = set(time_horizons + list(range(x[0], x[1])))

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

    for h in time_horizons:
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

        for f in features:
            if not is_numeric_dtype(forecast_df[f].dtype):
                try:
                    forecast_df[f] = forecast_df[f].astype(float)
                except:
                    raise ValueError('{} could not be converted to float. Please convert to numeric or encode with "encode_features: {}"'.format(f, f))

        if "link_function" in experiment_definition:
            if experiment_definition["link_function"] == 'log':
                forecast_df[
                    "{}_h_{}_pred".format(experiment_definition["target"], h)
                ] = da.expm1(fit_model.predict(forecast_df[features].to_dask_array(lengths=True)))
            sys.stdout.write("Forecasts made for horizon {}\n".format(h))
        else:
            forecast_df[
                "{}_h_{}_pred".format(experiment_definition["target"], h)
            ] = fit_model.predict(forecast_df[features].to_dask_array(lengths=True))
            sys.stdout.write("Forecasts made for horizon {}\n".format(h))

        factor_df = dd.from_array(forecast_df[features].to_dask_array(lengths=True) * da.from_array(fit_model.coef_))
        factor_df.columns = ["factor_{}".format(c) for c in features]
        for c in factor_df:
            forecast_df[c] = factor_df[c]

        if "confidence_intervals" in experiment_definition:
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
                        df['{}_h_{}_pred_b_{}'.format(experiment_definition["target"], h,
                                                      path.split("-")[-1])] = da.expm1(
                            bootstrap_model.predict(
                                dd.from_pandas(df[bootstrap_features], chunksize=10000).to_dask_array(
                                    lengths=True)))
                    else:
                        df['{}_h_{}_pred_b_{}'.format(experiment_definition["target"], h,
                                                      path.split("-")[-1])] = bootstrap_model.predict(
                            dd.from_pandas(df[bootstrap_features], chunksize=10000).to_dask_array(lengths=True))

                return df

            if "link_function" in experiment_definition:
                forecast_df = forecast_df.map_partitions(partial(load_and_predict_bootstrap_model,
                                                                 bootstrap_model_paths,
                                                                 experiment_definition['link_function']))

            else:
                forecast_df = forecast_df.map_partitions(partial(load_and_predict_bootstrap_model,
                                                                 bootstrap_model_paths,
                                                                 None))

            df_interval = dd.from_array(dd.from_array(
                forecast_df[['{}_h_{}_pred_b_{}'.format(experiment_definition["target"], h, i.split("-")[-1]) for i in
                             bootstrap_model_paths] + [
                                '{}_h_{}_pred'.format(experiment_definition["target"],
                                                      h)]].to_dask_array(lengths=True).T).repartition(
                npartitions=forecast_df.npartitions).quantile(
                [i * .01 for i in experiment_definition['confidence_intervals']]).to_dask_array(lengths=True).T).persist()
            df_interval.columns = ['{}_h_{}_pred_c_{}'.format(experiment_definition["target"], h, c) for c in
                                   experiment_definition["confidence_intervals"]]

            df_interval = df_interval.repartition(divisions=forecast_df.divisions).reset_index(drop=True)
            forecast_df = forecast_df.reset_index(drop=True).join(df_interval)

        forecast_df[experiment_definition["time_index"]] = dd.to_datetime(forecast_df[experiment_definition["time_index"]])

        forecast_df = forecast_df.sort_values(experiment_definition["time_index"])

        dd.to_parquet(
            forecast_df,
            "{}/forecast".format(
                write_path,
            )
        )
