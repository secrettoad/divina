import os
import pyspark
import pyspark.sql.functions as F
import json
import dask.dataframe as dd
import importlib
import pandas as pd
from .dataset import get_dataset


def dask_validate(vision_definition, divina_directory, vision_id, dask_client):

    def get_metrics(vision_definition, df, s):
        metrics = {'time_horizons': {}}
        for h in vision_definition['time_horizons']:
            metrics['time_horizons'][h] = {}
            df['resid_h_{}'.format(h)] = df[vision_definition['target']].shift(-h) - df[
                                     '{}_h_{}_pred'.format(vision_definition['target'], h)]
            metrics['time_horizons'][h]['mae'] = df[dd.to_datetime(df[vision_definition['time_index']], unit='s') > s]['resid_h_{}'.format(h)].abs().mean().compute()
        return metrics

    s3fs = importlib.import_module('s3fs')
    s3_fs = s3fs.S3FileSystem()

    metrics = {'splits': {}}
    for s in vision_definition['time_validation_splits']:

        df_pred = dd.read_parquet(
            "{}/coysu-divina-prototype-{}/predictions/s-{}".format(divina_directory, vision_id, pd.to_datetime(s).strftime("%Y%m%d-%H%M%S")))

        df_base, profile_base = get_dataset(vision_definition)
        df_base = df_base[[vision_definition['target'], vision_definition['time_index']]]

        if 'signal_dimensions' in vision_definition:
            df = df_pred.merge(df_base, on=[vision_definition['time_index']] + vision_definition['signal_dimensions'])
        else:
            df = df_pred.merge(df_base, on=[vision_definition['time_index']])
        metrics['splits'][s] = get_metrics(vision_definition, df, s)

        with s3_fs.open("{}/coysu-divina-prototype-{}/metrics.json".format(divina_directory, vision_id), 'w') as f:
            json.dump(metrics, f)
