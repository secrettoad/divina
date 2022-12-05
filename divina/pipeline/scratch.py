from utils import testClass, test_decorator
from prefect import flow, task
import dask.dataframe as dd
import pandas as pd

@test_decorator
def my_task(x, test_input):
    x.value += 1
    print(test_input)
    return x

@task()
def my_sum(x):
    return sum([i.value for i in x])

@flow(name='test')
def my_flow():
    my_tasks = []
    for t in [testClass(1), testClass(2), testClass(3)]:
        my_tasks.append(my_task(t))
    result = my_sum(my_tasks)
    df = dd.from_pandas(pd.DataFrame(columns=['a', 'b', 'c'], data=[[1, 2, 3], [4, 5, 6]]), npartitions=2)
    df.to_parquet('s3://test-bucket/test', storage_options={'client_kwargs':{'endpoint_url':'http://10.244.0.22'}})
    print(result)

my_flow()

###TODO - check out built-in task and flow result storage and kubernetes infrastructure functionality https://docs.prefect.io/concepts/infrastructure/
###TODO - also make sure that localexecutor works for debugging
### TODO  - move logic here
###TODO will eventually deploy via GKE and expose machine types there - use prefect kubernetesjob to achieve that along with GKE sdk
'''https://marclamberti.com/blog/airflow-on-kubernetes-get-started-in-10-mins/
### https://stackoverflow.com/questions/66160780/first-time-login-to-apache-airflow-asks-for-username-and-password-what-is-the-u
### helm repo add apache-airflow https://airflow.apache.org
### helm repo update
### helm search repo airflow
### kubectl create namespace airflow
### helm install airflow apache-airflow/airflow --namespace airflow --debug
### airflow users  create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin'''
### kubectl run --image=quay.io/minio/minio minio -n minio -- server /data --console-address ":9001"
### kubectl run --image=prefecthq/prefect:2.0.4-python3.9 prefect -n prefect -- prefect orion start

###TODO test if you can import local modules as well

