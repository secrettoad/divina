import dask.dataframe as dd
import pandas as pd
import pathlib

sales_df = pd.read_csv('s3://divina-quickstart/retail/sales/train.csv')
sales_df['StateHoliday'] = sales_df['StateHoliday'].astype('category').cat.codes
dd.from_pandas(sales_df, npartitions=100).to_parquet(pathlib.Path(pathlib.Path(__file__).parent.parent, 'raw', 'retail_sales', 'sales'))

store_df = pd.read_csv('s3://divina-quickstart/retail/store/store.csv')

dd.from_pandas(store_df, npartitions=100).to_parquet(pathlib.Path(pathlib.Path(__file__).parent.parent, 'raw', 'retail_sales', 'store'))

sales_df['Date'] = pd.to_datetime(sales_df['Date'])

sales_df['StateHoliday'] = sales_df['StateHoliday'].astype('category').cat.codes

sales_df = sales_df.set_index('Store').join(store_df.set_index('Store')).reset_index()

dd.from_pandas(sales_df, npartitions=100).to_parquet(pathlib.Path(pathlib.Path(__file__).parent.parent, 'datasets', 'retail_sales'))

pass

