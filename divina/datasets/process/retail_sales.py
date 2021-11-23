import dask.dataframe as dd
import pandas as pd
import pathlib

sales_df = pd.read_parquet(pathlib.Path(pathlib.Path(__file__).parent.parent, 'raw', 'retail_sales', 'sales'))
sales_df['StateHoliday'] = sales_df['StateHoliday'].astype('category').cat.codes
#dd.from_pandas(sales_df, npartitions=100).to_parquet(pathlib.Path(pathlib.Path(__file__).parent.parent, 'raw', 'retail_sales', 'sales'))

store_df = pd.read_parquet(pathlib.Path(pathlib.Path(__file__).parent.parent, 'raw', 'retail_sales', 'store'))

#dd.from_pandas(store_df, npartitions=100).to_parquet(pathlib.Path(pathlib.Path(__file__).parent.parent, 'raw', 'retail_sales', 'store'))

sales_df['Date'] = pd.to_datetime(sales_df['Date'])

sales_df['StateHoliday'] = sales_df['StateHoliday'].astype('category').cat.codes

sales_df = sales_df.set_index('Store').join(store_df.set_index('Store')).reset_index()
sales_df = sales_df[sales_df['Store'].isin(range(1, 4))]

dd.from_pandas(sales_df, npartitions=6).to_parquet(pathlib.Path(pathlib.Path(__file__).parent.parent, 'datasets', 'retail_sales'), schema='infer')

pass

