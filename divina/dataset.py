import dask.dataframe as dd
import backoff
from botocore.exceptions import ClientError
import pandas as pd
from .utils import cull_empty_partitions
from dask_ml.preprocessing import Categorizer, DummyEncoder
from sklearn.pipeline import make_pipeline
import numpy as np
from itertools import product
from .datasets.load import _load
from pandas.api.types import is_numeric_dtype


@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def _get_dataset(experiment_definition, start=None, end=None, pad=False):
    if experiment_definition["data_path"].startswith("divina://"):
        df = _load(experiment_definition["data_path"])
    else:
        df = dd.read_parquet("{}/*".format(experiment_definition["data_path"]))
    npartitions = (df.memory_usage(deep=True).sum().compute() // 104857600) + 1

    df[experiment_definition["time_index"]] = dd.to_datetime(df[experiment_definition["time_index"]])

    time_min, time_max = (
        df[experiment_definition["time_index"]].min().compute(),
        df[experiment_definition["time_index"]].max().compute(),
    )

    if "target_dimensions" in experiment_definition:
        df = df.groupby([experiment_definition["time_index"]] + experiment_definition["target_dimensions"]).agg(
            {**{c: "sum" for c in df.columns if
                df[c].dtype in [int, float] and c != experiment_definition['time_index']},
             **{c: "first" for c in df.columns if
                df[c].dtype not in [int, float] and c != experiment_definition['time_index']}}).drop(
            columns=experiment_definition["target_dimensions"]).reset_index()
    else:
        df = df.groupby(experiment_definition["time_index"]).agg(
            {**{c: "sum" for c in df.columns if
                df[c].dtype in [int, float] and c != experiment_definition['time_index']},
             **{c: "first" for c in df.columns if
                df[c].dtype not in [int, float] and c != experiment_definition['time_index']}}).reset_index()

    if start:
        if pd.to_datetime(start) < time_min:
            raise Exception(
                "Bad Start: {} < {} Check Dataset Time Range".format(start, time_min))
        else:
            df = df[dd.to_datetime(df[experiment_definition["time_index"]]) >= start]
            time_min = pd.to_datetime(str(start))

    if end:
        if pd.to_datetime(end) > time_max:
            if not "scenarios" in experiment_definition:
                raise Exception(
                    "Bad End: {} | {} Check Dataset Time Range".format(end, time_max))
        else:
            df = df[dd.to_datetime(df[experiment_definition["time_index"]]) <= end]
            time_max = pd.to_datetime(str(end))

    if "scenarios" in experiment_definition:
        if not "frequency" in experiment_definition:
            raise Exception(
                'Frequency of time series must be supplied. Please supply with "frequency: "D", "M", "s", etc."')
        if end:
            if start and pd.to_datetime(start) > time_max:
                new_dates = pd.date_range(pd.to_datetime(str(start)), pd.to_datetime(str(end)),
                                          freq=experiment_definition["frequency"])
            else:
                new_dates = pd.date_range(
                    time_max + pd.tseries.frequencies.to_offset(experiment_definition["frequency"]),
                    pd.to_datetime(str(end)),
                    freq=experiment_definition["frequency"])
            if len(new_dates) > 0:

                combinations = list(new_dates)
                if "target_dimensions" in experiment_definition:
                    combinations = [list(x) for x in product(combinations,
                                                             *[df[s].unique().compute().values for s in
                                                               experiment_definition["target_dimensions"]])]
                    scenario_columns = [experiment_definition["time_index"]] + experiment_definition[
                        "target_dimensions"]
                else:
                    combinations = [[x] for x in combinations]
                    scenario_columns = [experiment_definition["time_index"]]
                constant_columns = [c for c in experiment_definition["scenarios"] if
                                    experiment_definition["scenarios"][c]["mode"] == "constant"]
                for c in constant_columns:
                    combinations = [x[0] + [x[1]] for x in
                                    product(combinations, experiment_definition["scenarios"][c]["constant_values"])]
                df_scenario = dd.from_pandas(pd.DataFrame(combinations, columns=scenario_columns + constant_columns),
                                             npartitions=npartitions)
                last_columns = [c for c in experiment_definition["scenarios"] if
                                experiment_definition["scenarios"][c]["mode"] == "last"]
                if len(last_columns) > 0:
                    if "target_dimensions" in experiment_definition:
                        last = df.groupby(experiment_definition["target_dimensions"])[last_columns].last().compute()
                        meta = df_scenario.join(last.reset_index(drop=True), how="right")
                        df_scenario = df_scenario.groupby(experiment_definition["target_dimensions"]).apply(
                            lambda x: x.set_index(experiment_definition["target_dimensions"]).join(
                                last).reset_index().set_index(experiment_definition["time_index"]).reset_index(),
                            meta=meta).reset_index(drop=True)
                    else:
                        last = df[last_columns].tail(1)
                        for l in last_columns:
                            df_scenario[l] = last[l]

                df = dd.concat([df.set_index(experiment_definition["time_index"]), df_scenario.set_index(experiment_definition["time_index"])], axis=0).reset_index()

    if "joins" in experiment_definition:
        for i, join in enumerate(experiment_definition["joins"]):
            try:
                if join["data_path"].startswith("divina://"):
                    join_df = _load(join["data_path"])
                else:
                    join_df = dd.read_parquet("{}/*".format(join["data_path"]))
            except IndexError:
                raise Exception("Could not load dataset {}. No parquet files found.".format(join["data_path"]))
            join_df[join["join_on"][0]] = join_df[join["join_on"][0]].astype(df[join["join_on"][1]].dtype)
            df = df.merge(
                join_df,
                how="left",
                left_on=join["join_on"][0],
                right_on=join["join_on"][1],
                suffixes=("", "{}_".format(join["as"])),
            )

    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(float)

    if "include_features" in experiment_definition:
        df = df[[experiment_definition["target"], experiment_definition["time_index"]] + experiment_definition[
            "include_features"]]

    if "bin_features" in experiment_definition:
        for c in experiment_definition["bin_features"]:
            edges = [-np.inf] + experiment_definition["bin_features"][c] + [np.inf]
            for v, v_1 in zip(edges, edges[1:]):
                df["{}_({}, {}]".format(c, v, v_1)] = 1
                df["{}_({}, {}]".format(c, v, v_1)] = df["{}_({}, {}]".format(c, v, v_1)].where(
                    ((df[c] < v_1) & (df[c] >= v)), 0)

    if "encode_features" in experiment_definition:
        for c in experiment_definition["encode_features"]:
            if df[c].dtype == int:
                df[c] = df[c].astype(float)
            else:
                df[c] = df[c]
            df["{}_dummy".format(c)] = df[c]

        pipe = make_pipeline(
            Categorizer(columns=experiment_definition["encode_features"]),
            DummyEncoder(columns=experiment_definition["encode_features"]))

        pipe.fit(df)

        df = pipe.transform(df)

        for c in experiment_definition["encode_features"]:
            df[c] = df["{}_dummy".format(c)]
        df = df.drop(columns=["{}_dummy".format(c) for c in experiment_definition["encode_features"]])

    if "interaction_features" in experiment_definition:
        for t in experiment_definition["interaction_features"]:
            if t in experiment_definition["encode_features"]:
                pipe = make_pipeline(
                    Categorizer(columns=[t]),
                    DummyEncoder(columns=[t]))
                interactions = list(pipe.fit(df[[t]]).steps[1][1].transformed_columns_)
            else:
                interactions = [t]
            for c in interactions:
                for w in experiment_definition["interaction_features"][t]:
                    if w in experiment_definition["encode_features"]:
                        pipe = make_pipeline(
                            Categorizer(columns=[w]),
                            DummyEncoder(columns=[w]))
                        v = list(pipe.fit(df[[w]]).steps[1][1].transformed_columns_)
                    else:
                        v = [w]
                    for m in v:
                        if not '{}-x-{}'.format(c, m) in df.columns:
                            if not all([is_numeric_dtype(x) for x in df[[t, m]].dtypes]):
                                df['{}-x-{}'.format(c, m)] = df[t].astype(str) + "_*_" + df[m].astype(str)
                            else:
                                df['{}-x-{}'.format(c, m)] = df[t] * df[m]

    if "encode_features" in experiment_definition:
        df = df.drop(columns=experiment_definition["encode_features"])

    if "drop_features" in experiment_definition:
        df = df.drop(columns=experiment_definition["drop_features"])

    df[experiment_definition["time_index"]] = dd.to_datetime(df[experiment_definition["time_index"]])
    df = df.repartition(npartitions=npartitions)
    df = cull_empty_partitions(df)
    df['index'] = 1
    df['index'] = df['index'].cumsum()
    df['index'] = df['index'] - 1
    df = df.set_index('index')
    df.index.name = None
    return df.copy().persist()
