from dask_kubernetes.operator import KubeCluster
cluster = KubeCluster(name="my_dask_cluster", image='ghcr.io/dask/dask:latest')
cluster.scale(10)

### helm repo add dask https://helm.dask.org
### helm repo update
#### helm install --create-namespace -n dask-operator ./dask-kubernetes-operator/ -f dask-kubernetes-operator/values.yaml --generate-name

