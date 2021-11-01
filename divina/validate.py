import json
import dask.dataframe as dd
import pandas as pd
from .dataset import _get_dataset
import os
import backoff
from botocore.exceptions import ClientError
from .utils import cull_empty_partitions, validate_forecast_definition
import joblib
from functools import partial
import dask.array as da
import sys


@validate_forecast_definition
@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def _validate(s3_fs, forecast_definition, write_path, read_path):
    def get_metrics(forecast_definition, df):
        metrics = {"time_horizons": {}}
        for h in forecast_definition["time_horizons"]:
            metrics["time_horizons"][h] = {}
            df["resid_h_{}".format(h)] = (
                    df[forecast_definition["target"]].shift(-h)
                    - df["{}_h_{}_pred".format(forecast_definition["target"], h)]
            )
            metrics["time_horizons"][h]["mae"] = (
                df[
                    "resid_h_{}".format(h)
                ]
                    .abs()
                    .mean()
                    .compute()
            )
        return metrics

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
        read_ls = s3_fs.ls
        write_open = s3_fs.open
        bootstrap_prefix = None
    else:
        read_open = open
        read_ls = os.listdir
        write_open = open
        bootstrap_prefix = os.path.join(read_path, 'models', 'bootstrap')

    dataset_kwargs = {}
    for k in ['validate_start', 'validate_end']:
        if k in forecast_definition:
            dataset_kwargs.update({k.split('_')[1]: forecast_definition[k]})

    df = _get_dataset(forecast_definition)

    metrics = {"splits": {}}

    horizon_ranges = [x for x in forecast_definition["time_horizons"] if type(x) == tuple]
    if len(horizon_ranges) > 0:
        forecast_definition["time_horizons"] = [x for x in forecast_definition["time_horizons"] if type(x) == int]
        for x in horizon_ranges:
            forecast_definition["time_horizons"] = set(forecast_definition["time_horizons"] + list(range(x[0], x[1])))

    df = df[
        [forecast_definition["target"], forecast_definition["time_index"]]
    ]

    time_min, time_max = (
        df[forecast_definition["time_index"]].min().compute(),
        df[forecast_definition["time_index"]].max().compute(),
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
        validate_df = _get_dataset(forecast_definition, pad=False, **validate_kwargs)

        for h in forecast_definition["time_horizons"]:
            with read_open(
                    "{}/models/s-{}_h-{}".format(
                        read_path,
                        pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
                        h,
                    ),
                    "rb",
            ) as f:
                fit_model = joblib.load(f)
            with read_open(
                    "{}/models/s-{}_h-{}_params.json".format(
                        read_path,
                        pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
                        h,
                    ),
                    "r",
            ) as f:
                fit_model_params = json.load(f)
            features = fit_model_params["features"]
            for f in features:
                if not f in validate_df.columns:
                    validate_df[f] = 0

            if "link_function" in forecast_definition:
                if forecast_definition["link_function"] == 'log':
                    validate_df[
                        "{}_h_{}_pred".format(forecast_definition["target"], h)
                    ] = da.expm1(fit_model.predict(validate_df[features].to_dask_array(lengths=True)))
                    sys.stdout.write("Validation predictions made for split {}\n".format(s))

            else:
                validate_df[
                    "{}_h_{}_pred".format(forecast_definition["target"], h)
                ] = fit_model.predict(validate_df[features].to_dask_array(lengths=True))
                sys.stdout.write("Validation predictions made for split {}\n".format(s))

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
                            bootstrap_features = bootstrap_params['features']
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
                    validate_df = validate_df.map_partitions(partial(load_and_predict_bootstrap_model,
                                                                     bootstrap_model_paths,
                                                                     forecast_definition['confidence_intervals'],
                                                                     forecast_definition['link_function']))
                else:
                    validate_df = validate_df.map_partitions(partial(load_and_predict_bootstrap_model,
                                                                     bootstrap_model_paths,
                                                                     forecast_definition['confidence_intervals'],
                                                                     None))

            if not pd.to_datetime(str(time_min)) < pd.to_datetime(str(s)) < pd.to_datetime(str(time_max)):
                raise Exception("Bad Validation Split: {} | Check Dataset Time Range".format(s))

            validate_df = cull_empty_partitions(validate_df)
            metrics["splits"][s] = get_metrics(forecast_definition, validate_df)

            with write_open("{}/metrics.json".format(write_path), "w") as f:
                json.dump(metrics, f)

        validate_df[forecast_definition["time_index"]] = dd.to_datetime(validate_df[forecast_definition["time_index"]])

        dd.to_parquet(
            validate_df[
                [forecast_definition["time_index"]]
                + [
                    "{}_h_{}_pred".format(forecast_definition["target"], h)
                    for h in forecast_definition["time_horizons"]
                ]
                ],
            "{}/validation/s-{}".format(
                write_path,
                pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
            )
        )
