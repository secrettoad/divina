import sys
import dask.dataframe as dd
import pandas as pd
import importlib
import joblib
import pathlib
from .dataset import get_dataset


def dask_predict(s3_fs, dask_client, vision_definition, vision_id, divina_directory):

    df, profile = get_dataset(vision_definition)

    if 'drop_features' in vision_definition:
        df = df.drop(columns=vision_definition['drop_features'])

    if not divina_directory[:5] == 's3://':
        pathlib.Path("{}/{}/predictions".format(divina_directory, vision_id
                                                                       )).mkdir(parents=True, exist_ok=True)

    for s in vision_definition['time_validation_splits']:

        for h in vision_definition['time_horizons']:
            with s3_fs.open("{}/{}/models/s-{}_h-{}".format(divina_directory,
                                                                                   vision_id,
                                                                                   pd.to_datetime(str(s)).strftime(
                                                                                       "%Y%m%d-%H%M%S"), h), 'rb') as f:
                fit_model = joblib.load(f)

            df['{}_h_{}_pred'.format(vision_definition['target'], h)] = fit_model.predict(df[[c for c in df.columns
                                                                                                 if not c in [
                    vision_definition['time_index'], vision_definition['target']] + ['{}_h_{}'.format(vision_definition['target'], h) for h in
                                          vision_definition['time_horizons']]]].to_dask_array(lengths=True))

            sys.stdout.write('Predictions made for horizon {}\n'.format(h))

        dd.to_parquet(df[[vision_definition['time_index']] + ['{}_h_{}_pred'.format(vision_definition['target'], h) for h in
                                          vision_definition['time_horizons']]], "{}/{}/predictions/s-{}".format(divina_directory, vision_id,
                                                                               pd.to_datetime(str(s)).strftime(
                                                                                   "%Y%m%d-%H%M%S")))
