<div align="center">
  <img src="https://storage.googleapis.com/coysuweb-static/assets/images/logo/3.png"><br>
</div>

-----------------

# divina: scalable and hyper-interpretable causal forecasting toolkit
[![Continuous Integration](https://github.com/secrettoad/divina/actions/workflows/prod.yaml/badge.svg)](https://github.com/secrettoad/divina/actions/workflows/prod.yaml)
[![PyPI Latest Release](https://img.shields.io/pypi/v/divina.svg)](https://pypi.org/project/divina/)
[![Package Status](https://img.shields.io/pypi/status/divina.svg)](https://pypi.org/project/divina/)
[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://github.com/secrettoad/divina/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## What is it?

`divina` confronts four main problems for professional forecasters:
  
  - Multiple-horizon forecasting often involves repeated programming of the same, complex code across train, predict, validation and visualization
  - Many forecasting model implementations do not follow the standard scikit-interface or scale well to large datasets
  - Different forecasting models often benefit from the complex engineering of the same time-sensitive features
  - Because of the above three, deployment and scaling of multi-horizon forecasting ensembles is considerably more complex than a typical machine learning pipeline


## Main Features
`divina` addresses the aforementioned problems by:

  - Providing a single Python object with a simple interface that abstracts away the complexities of multi-horizon train, predict, valiation and visualization
  - Providing a library of interface-standardized model ensemble candidates that can be mixed and matched depending on the forecasting problem
  - Providing consistent, efficient implementations of popular time-series engineered features 
  - Built-in integration with Dask for efficient, cloud-based scaling and Prefect for automation, fault-tolerance, queue management and artifact persistence 
  
  [interpretation]: https://github.com/secrettoad/divina
  
## Roadmap
Current development priorities and improvements slated for next and beta release are:

  - Addition of visualization methods that produce commonly-required charts via Highcharts
  - Additional machine learning model options, such as XGBoost,
  - Additional boosting model options, such as RNNs, LSTMs, ARIMA, SARIMA, etc.
  - Addition of more realistic test cases, useful error messages and robust documentation 
  - Addition of GPU support via CUDA, CUDF and CUML

## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/secrettoad/divina

## Documentation
``divina``'s documentation is available [here](https://secrettoad.github.io/divina/#). 

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/divina)

```sh
pip install divina
```

## Dependencies
- [dask - Adds support for arbitrarily large datasets via remote, parallelized compute](https://www.dask.org)
- [dask-ml - Provides distributed-optimized implementations of many popular models](https://ml.dask.org)
- [s3fs - Allows for easy and efficient access to S3](https://github.com/dask/s3fs)
- [pyarrow - Enables persistence of datasets as storage and compute efficent parquet files](https://arrow.apache.org/docs/python/)
- [prefect - Enables task orchestration, tracking and persistence](https://prefect.io)


## Testing
For local integration testing, run the following commands in order to create the necessary Prefect and Min.io containers.
```sh
docker pull jhurdle/divina-storage
docker pull jhurdle/divina-prefect
docker run -p 9000:9000 jhurdle/divina-storage 
docker run -p 4200:4200 jhurdle/divina-prefect 
pytest divina/divina/tests
```

## License
[AGPL](LICENSE)

## Background
Work on ``divina`` started at [Coysu Consulting](https://www.coysu.com/) (a technology consulting firm) in 2020 and
has been under active development since then.

## Getting Help
For usage questions, the best place to go to is [StackOverflow](https://stackoverflow.com/questions/tagged/divina).

## Discussion and Development
Most development discussions take place on GitHub in this repo.

## Contributing to divina 

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

If you are simply looking to start working with the divina codebase, navigate to the [GitHub "issues" tab](https://github.com/secrettoad/divina/issues) and start looking through interesting issues.

