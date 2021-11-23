import dask.dataframe as dd
import pandas as pd
import datetime
import pathlib

time_df = dd.read_parquet('s3://divina-quickstart/time2/data').compute()

time_df.columns = [c.replace('_', ' ').title().replace(' ', '') for c in time_df]
time_df['Date'] = pd.to_datetime(time_df['Date'])

time_df['LastDayOfMonth'] = 0

time_df['LastDayOfMonth'][(time_df['Date'] + datetime.timedelta(days=1)).dt.day == 1] = 1

time_df['DayOfMonth'] = time_df['Date'].dt.day

time_df['WeekOfYear'] = time_df['Date'].dt.week

dd.from_pandas(time_df, npartitions=20).to_parquet(pathlib.Path(pathlib.Path(__file__).parent.parent, 'datasets', 'time'))

pass

