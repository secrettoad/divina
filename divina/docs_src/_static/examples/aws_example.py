import os

import dask.dataframe as dd
import pandas as pd

from divina import Divina
from divina.divina.pipeline.utils import DaskConfiguration

os.environ['AWS_ACCESS_KEY_ID'] = 'your access key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your secret key'

dask_configuration = DaskConfiguration(destination='aws', num_workers=3)

example_data = pd.DataFrame(
    data=[
        ["2011-01-01", 3, 6],
        ["2011-01-02", 2, 4],
        ["2011-01-03", 8, 6],
        ["2011-01-04", 1, 1],
        ["2011-01-05", 2, 3],
    ],
    columns=["a", "b", "c"],
)

example_data_dask = dd.from_pandas(example_data, npartitions=1)

example_pipeline = Divina(target="c", time_index="a", frequency="D")

y_hat_insample = example_pipeline.fit(
    example_data_dask,
    dask_configuration=dask_configuration)[0].causal_validation.predictions

y_hat_out_of_sample = example_pipeline.predict(
    example_data_dask.drop(columns="c"),
    dask_configuration=dask_configuration
).causal_predictions.predictions
