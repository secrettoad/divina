import dask.dataframe as dd
import backoff
from botocore.exceptions import ClientError
import pandas as pd
from .utils import cull_empty_partitions
from dask_ml.preprocessing import Categorizer, DummyEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np


def chunk(s):
    # for the comments, assume only a single grouping column, the
    # implementation can handle multiple group columns.
    #
    # s is a grouped series. value_counts creates a multi-series like
    # (group, value): count
    return s.value_counts()


def agg(s):
    return s.apply(lambda s: s.groupby(level=-1).sum())


def finalize(s):
    # s is a multi-index series of the form (group, value): count. First
    # manually group on the group part of the index. The lambda will receive a
    # sub-series with multi index. Next, drop the group part from the index.
    # Finally, determine the index with the maximum value, i.e., the mode.
    level = list(range(s.index.nlevels - 1))
    return (
        s.groupby(level=level)
            .apply(lambda s: s.reset_index(level=level, drop=True).argmax())
    )


mode = dd.Aggregation('mode', chunk, agg, finalize)


@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def get_dataset(forecast_definition, start=None, end=None, pad=False):
    df = dd.read_parquet("{}/*".format(forecast_definition["dataset_directory"]))

    time_min, time_max = (
        pd.to_datetime(str(df[forecast_definition["time_index"]].min().compute())),
        pd.to_datetime(str(df[forecast_definition["time_index"]].max().compute())),
    )

    if "signal_dimensions" in forecast_definition:
        df = df.groupby([forecast_definition["time_index"]] + forecast_definition["signal_dimensions"]).agg(
            {**{c: "sum" for c in df.columns if df[c].dtype in [int, float] and c != forecast_definition['time_index']},
             **{c: "first" for c in df.columns if
                df[c].dtype not in [int, float] and c != forecast_definition['time_index']}}).reset_index()
    else:

        df = df.groupby(forecast_definition["time_index"]).agg(
            {**{c: "sum" for c in df.columns if df[c].dtype in [int, float] and c != forecast_definition['time_index']},
             **{c: "first" for c in df.columns if
                df[c].dtype not in [int, float] and c != forecast_definition['time_index']}}).reset_index()

    if start:
        if pd.to_datetime(start) < time_min:
            raise Exception(
                "Bad Start: {} < {} Check Dataset Time Range".format(start, time_min))
        else:
            df = df[dd.to_datetime(df[forecast_definition["time_index"]]) >= start]
            time_min = pd.to_datetime(str(start))

    if end:
        if pd.to_datetime(end) > time_max:
            if pad:
                if pd.to_datetime(start) > time_max:
                    new_dates = pd.date_range(pd.to_datetime(str(start)), pd.to_datetime(str(end)),
                                              freq=forecast_definition["forecast_freq"])
                else:
                    new_dates = pd.date_range(time_max, pd.to_datetime(str(end)),
                                              freq=forecast_definition["forecast_freq"])
                new_dates_df = dd.from_pandas(pd.DataFrame(new_dates, columns=[forecast_definition["time_index"]]),
                                              chunksize=10000)
                df = df.append(new_dates_df)
            else:
                raise Exception(
                    "Bad End: {} | {} Check Dataset Time Range".format(end, time_max))
        else:
            df = df[dd.to_datetime(df[forecast_definition["time_index"]]) <= end]
        time_max = pd.to_datetime(str(end))

    if "joins" in forecast_definition:
        for i, join in enumerate(forecast_definition["joins"]):
            join_df = dd.read_parquet("{}/*".format(join["dataset_directory"]))
            join_df[join["join_on"][1]] = join_df[join["join_on"][1]].astype(df[join["join_on"][0]].dtype)
            df = df.merge(
                join_df,
                how="left",
                left_on=join["join_on"][0],
                right_on=join["join_on"][1],
                suffixes=("", "{}_".format(join["as"])),
            )

    if "drop_features" in forecast_definition:
        df = df.drop(columns=forecast_definition["drop_features"])

    if "scenarios" in forecast_definition:
        for x in forecast_definition["scenarios"]:

            scenario_ranges = [x for x in forecast_definition["scenarios"][x]["values"] if type(x) == list]
            if len(scenario_ranges) > 0:
                forecast_definition["scenarios"][x]["values"] = [x for x in
                                                                 forecast_definition["scenarios"][x]["values"] if
                                                                 type(x) == int]
                for y in scenario_ranges:
                    forecast_definition["scenarios"][x]["values"] = set(
                        forecast_definition["scenarios"][x]["values"] + list(range(y[0], y[1] + 1)))

            df_scenario = df[(df[forecast_definition["time_index"]] <= pd.to_datetime(
                str(forecast_definition["scenarios"][x]["end"]))) & (
                                     df[forecast_definition["time_index"]] >= pd.to_datetime(
                                 str(forecast_definition["scenarios"][x]["start"])))]
            df = df[(df[forecast_definition["time_index"]] > pd.to_datetime(
                str(forecast_definition["scenarios"][x]["end"]))) | (
                            df[forecast_definition["time_index"]] < pd.to_datetime(
                        str(forecast_definition["scenarios"][x]["start"])))]
            for v in forecast_definition["scenarios"][x]["values"]:
                df_scenario[x] = v
                df = df.append(df_scenario)

    if "encode_features" in forecast_definition:
        for c in forecast_definition["encode_features"]:
            df['dummy_{}'.format(c)] = df[c]

        pipe = make_pipeline(
            Categorizer(columns=forecast_definition["encode_features"]),
            DummyEncoder(columns=forecast_definition["encode_features"]))

        pipe.fit(df)

        df = pipe.transform(df)

        df = df.drop(columns=['dummy_{}'.format(c) for c in forecast_definition["encode_features"]])

    if "bin_features" in forecast_definition:
        for c in forecast_definition["bin_features"]:
            edges = [-np.inf] + forecast_definition["bin_features"][c] + [np.inf]
            for v, v_1 in zip(edges, edges[1:]):
                df["{}_({}, {}]".format(c, v, v_1)] = 1
                df["{}_({}, {}]".format(c, v, v_1)] = df["{}_({}, {}]".format(c, v, v_1)].where(
                    ((df[c] < v_1) & (df[c] >= v)), 0)

    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)

    if "interaction_terms" in forecast_definition:
        for t in forecast_definition["interaction_terms"]:
            if forecast_definition["interaction_terms"][t] == '*':
                for c in [f for f in df.columns if not f in
                                                       [t, forecast_definition['target'],
                                                        forecast_definition['time_index']]]:
                    if not '{}-x-{}'.format(c, t) in df.columns:
                        df['{}-x-{}'.format(t, c)] = df[t] * df[c]
            else:
                for c in forecast_definition["interaction_terms"][t]:
                    df['{}-x-{}'.format(t, c)] = df[t] * df[c]

    df = cull_empty_partitions(df)
    return df
