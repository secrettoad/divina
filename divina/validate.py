import json
import dask.dataframe as dd
import pandas as pd
from .dataset import _get_dataset
import os
import backoff
from botocore.exceptions import ClientError
from .utils import cull_empty_partitions, validate_experiment_definition
import joblib
from functools import partial
import dask.array as da
import sys


@validate_experiment_definition
@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def _validate(s3_fs, experiment_definition, write_path, read_path):
    def get_metrics(experiment_definition, df, time_horizons):
        metrics = {"time_horizons": {}}
        for h in time_horizons:
            metrics["time_horizons"][h] = {}
            df["resid_h_{}".format(h)] = (
                    df[experiment_definition["target"]].shift(-h)
                    - df["{}_h_{}_pred".format(experiment_definition["target"], h)]
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
    if not "time_horizons" in experiment_definition:
        time_horizons = [0]
    else:
        time_horizons = experiment_definition["time_horizons"]
    if not "time_validation_splits" in experiment_definition:
        time_validation_splits = []
    else:
        time_validation_splits = experiment_definition["time_validation_splits"]

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
        if k in experiment_definition:
            dataset_kwargs.update({k.split('_')[1]: experiment_definition[k]})

    df = _get_dataset(experiment_definition)

    metrics = {"splits": {}}

    horizon_ranges = [x for x in time_horizons if type(x) == tuple]
    if len(horizon_ranges) > 0:
        time_horizons = [x for x in time_horizons if type(x) == int]
        for x in horizon_ranges:
            time_horizons = set(time_horizons + list(range(x[0], x[1])))

    df = df[
        [experiment_definition["target"], experiment_definition["time_index"]]
    ]

    time_min, time_max = (
        df[experiment_definition["time_index"]].min().compute(),
        df[experiment_definition["time_index"]].max().compute(),
    )

    for s in time_validation_splits:

        validate_kwargs = {}
        if 'validate_start' in experiment_definition:
            if pd.to_datetime(str(s)) < pd.to_datetime(str(experiment_definition["validate_start"])):
                validate_kwargs["start"] = pd.to_datetime(str(s))
            else:
                validate_kwargs["start"] = pd.to_datetime(str(experiment_definition["validate_start"]))
        else:
            validate_kwargs["start"] = pd.to_datetime(str(s))
        if 'validate_end' in experiment_definition:
            if pd.to_datetime(str(s)) > pd.to_datetime(str(experiment_definition["validate_end"])):
                raise Exception(
                    "Bad End: {} | Check Dataset Time Range".format(
                        pd.to_datetime(str(experiment_definition['forecast_start']))))
            else:
                validate_kwargs["end"] = pd.to_datetime(str(s))
        validate_df = _get_dataset(experiment_definition, pad=False, **validate_kwargs)

        for h in time_horizons:
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

            if "link_function" in experiment_definition:
                if experiment_definition["link_function"] == 'log':
                    validate_df[
                        "{}_h_{}_pred".format(experiment_definition["target"], h)
                    ] = da.expm1(fit_model.predict(validate_df[features].to_dask_array(lengths=True)))
                    sys.stdout.write("Validation predictions made for split {}\n".format(s))

            else:
                validate_df[
                    "{}_h_{}_pred".format(experiment_definition["target"], h)
                ] = fit_model.predict(validate_df[features].to_dask_array(lengths=True))
                sys.stdout.write("Validation predictions made for split {}\n".format(s))

            if "confidence_intervals" in experiment_definition:
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
                            df['{}_h_{}_pred_b_{}'.format(experiment_definition["target"], h, i)] = da.expm1(
                                bootstrap_model.predict(
                                    dd.from_pandas(df[bootstrap_features], chunksize=10000).to_dask_array(
                                        lengths=True)))
                        else:
                            df['{}_h_{}_pred_b_{}'.format(experiment_definition["target"], h,
                                                          i)] = bootstrap_model.predict(
                                dd.from_pandas(df[bootstrap_features], chunksize=10000).to_dask_array(lengths=True))

                    df_agg = df[['{}_h_{}_pred_b_{}'.format(experiment_definition["target"], h, i) for i in
                                 range(0, len(paths))] + ['{}_h_{}_pred'.format(experiment_definition["target"], h)]].T
                    for i in intervals:
                        if i > 50:
                            interpolation = 'higher'
                        elif i < 50:
                            interpolation = 'lower'
                        else:
                            interpolation = 'linear'
                        df['{}_h_{}_pred_c_{}'.format(experiment_definition["target"], h, i)] = df_agg.quantile(i * .01,
                                                                                                              interpolation=interpolation).T
                    return df

                if "link_function" in experiment_definition:
                    validate_df = validate_df.map_partitions(partial(load_and_predict_bootstrap_model,
                                                                     bootstrap_model_paths,
                                                                     experiment_definition['confidence_intervals'],
                                                                     experiment_definition['link_function']))
                else:
                    validate_df = validate_df.map_partitions(partial(load_and_predict_bootstrap_model,
                                                                     bootstrap_model_paths,
                                                                     experiment_definition['confidence_intervals'],
                                                                     None))

            if not pd.to_datetime(str(time_min)) < pd.to_datetime(str(s)) < pd.to_datetime(str(time_max)):
                raise Exception("Bad Validation Split: {} | Check Dataset Time Range".format(s))

            validate_df = cull_empty_partitions(validate_df)
            metrics["splits"][s] = get_metrics(experiment_definition, validate_df, time_horizons)

            with write_open("{}/metrics.json".format(write_path), "w") as f:
                json.dump(metrics, f)

        validate_df[experiment_definition["time_index"]] = dd.to_datetime(validate_df[experiment_definition["time_index"]])

        dd.to_parquet(
            validate_df[
                [experiment_definition["time_index"]]
                + [
                    "{}_h_{}_pred".format(experiment_definition["target"], h)
                    for h in time_horizons
                ]
                ],
            "{}/validation/s-{}".format(
                write_path,
                pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
            )
        )
