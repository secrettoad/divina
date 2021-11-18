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


@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def _get_dataset(experiment_definition, start=None, end=None, pad=False):
    if experiment_definition["data_path"].startswith("divina://"):
        df = _load(experiment_definition["data_path"])
    else:
        df = dd.read_parquet("{}/*".format(experiment_definition["data_path"]))
    df = df.reset_index()
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
            if pad:
                if not "scenario_freq" in experiment_definition:
                    raise Exception(
                        'Frequency of time series must be supplied. Please supply with "scenario_freq: "D", "M", "s", etc."')
                if start and pd.to_datetime(start) > time_max:
                    new_dates = pd.date_range(pd.to_datetime(str(start)), pd.to_datetime(str(end)),
                                              freq=experiment_definition["scenario_freq"])
                else:
                    new_dates = pd.date_range(time_max + pd.offsets.Day(), pd.to_datetime(str(end)),
                                              freq=experiment_definition["scenario_freq"])
                if "target_dimensions" in experiment_definition:

                    combinations = list(product(list(new_dates),
                                                *[df[s].unique().compute().values for s in
                                                  experiment_definition["target_dimensions"]]))
                    new_dates_df = dd.from_pandas(pd.DataFrame(combinations,
                                                               columns=[experiment_definition["time_index"]] +
                                                                       experiment_definition["target_dimensions"]),
                                                  npartitions=df.npartitions)

                else:
                    new_dates_df = dd.from_pandas(
                        pd.DataFrame(new_dates, columns=[experiment_definition["time_index"]]),
                        npartitions=df.npartitions)

                df = df.append(new_dates_df)
            else:
                raise Exception(
                    "Bad End: {} | {} Check Dataset Time Range".format(end, time_max))
        else:
            df = df[dd.to_datetime(df[experiment_definition["time_index"]]) <= end]
        time_max = pd.to_datetime(str(end))

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

    if "scenarios" in experiment_definition:
        for x in experiment_definition["scenarios"]:
            scenario_ranges = [c for c in x["values"] if type(c) == list]
            if len(scenario_ranges) > 0:
                x["values"] = [c for c in
                               x["values"] if
                               type(c) == int]
                for y in scenario_ranges:
                    x["values"] = set(
                        x["values"] + list(range(y[0], y[1] + 1)))

            df_scenario = df[(dd.to_datetime(df[experiment_definition["time_index"]]) <= pd.to_datetime(
                str(x["end"]))) & (
                                     dd.to_datetime(df[experiment_definition["time_index"]]) >= pd.to_datetime(
                                 str(x["start"])))]
            df = df[(dd.to_datetime(df[experiment_definition["time_index"]]) > pd.to_datetime(
                str(x["end"]))) | (
                            dd.to_datetime(df[experiment_definition["time_index"]]) < pd.to_datetime(
                        str(x["start"])))]
            for v in x["values"]:
                df_scenario[x["feature"]] = v
                df = df.append(df_scenario)

    if "bin_features" in experiment_definition:
        for c in experiment_definition["bin_features"]:
            edges = [-np.inf] + experiment_definition["bin_features"][c] + [np.inf]
            for v, v_1 in zip(edges, edges[1:]):
                df["{}_({}, {}]".format(c, v, v_1)] = 1
                df["{}_({}, {}]".format(c, v, v_1)] = df["{}_({}, {}]".format(c, v, v_1)].where(
                    ((df[c] < v_1) & (df[c] >= v)), 0)

    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(float)

    if "drop_features" in experiment_definition:
        df = df.drop(columns=experiment_definition["drop_features"])

    elif "include_features" in experiment_definition:
        df = df[[experiment_definition["target"], experiment_definition["time_index"]] + experiment_definition[
            "include_features"]]

    if "encode_features" in experiment_definition:
        for c in experiment_definition["encode_features"]:
            if df[c].dtype == int:
                df[c] = df[c].astype(float)
            else:
                df[c] = df[c]

        pipe = make_pipeline(
            Categorizer(columns=experiment_definition["encode_features"]),
            DummyEncoder(columns=experiment_definition["encode_features"]))

        pipe.fit(df)

        df = pipe.transform(df)

    if "interaction_terms" in experiment_definition:
        for t in experiment_definition["interaction_terms"]:
            if experiment_definition["interaction_terms"][t] == '*':
                for c in [f for f in df.columns if not f in
                                                       [t, experiment_definition['target'],
                                                        experiment_definition['time_index']]]:
                    if not '{}-x-{}'.format(c, t) in df.columns:
                        df['{}-x-{}'.format(t, c)] = df[t] + df[c]
            else:
                for c in experiment_definition["interaction_terms"][t]:
                    df['{}-x-{}'.format(t, c)] = df[t] + df[c]

    df[experiment_definition["time_index"]] = dd.to_datetime(df[experiment_definition["time_index"]])
    df = df.repartition(npartitions=npartitions)
    df = cull_empty_partitions(df)
    df['index'] = 1
    df['index'] = df['index'].cumsum()
    df['index'] = df['index'] - 1
    df = df.set_index('index')
    df.index.name = None
    return df.persist()
