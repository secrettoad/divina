from divina.divina.pipeline.pipeline import Pipeline
import pandas as pd
import dask.dataframe as dd


from prefect import flow
@flow(
    name='divina_example_pipeline', persist_result=True
)
def run_pipeline(pipeline, data):
    pipeline.fit(data, prefect=True)
    return example_pipeline.predict(data.drop(columns=pipeline.target), prefect=True)

example_data = pd.DataFrame(
    data=[['2011-01-01', 3, 6],
          ['2011-01-02', 2, 4],
          ['2011-01-03', 8, 6],
          ['2011-01-04', 1, 1],
          ['2011-01-05', 2, 3]], columns=['a', 'b', 'c'])

example_data_dask = dd.from_pandas(example_data, npartitions=1)

example_pipeline = Pipeline(target="c",
                                 time_index="a",
                                 frequency="D")

print(run_pipeline(example_pipeline, example_data_dask))