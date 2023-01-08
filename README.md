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

`divina` is essentially a convenience wrapper that facilitates training, prediction, validation and deployment of an ensemble consisting of a causal, interpretable model that is boosted by an endogenous time-series model, allowing for high levels of automation and accuracy while still emphasizing and relying on the causal relationships discovered by the user. This ensemble structure is delivered with swappable model types to be able to suit many different kinds of forecasting problems. :mod:`divina` is also fully integrated with both `dask` and `prefect` meaning that distributed compute and pipeline orchestration can be enabled with the flip of a switch. For more information of `divina`'s features, check out the [documentation](https://secrettoad.github.io/divina/#).


## Main Features
Here are just a few of the things that divina does well:

  - Abstraction of all necessary configuration of a pipeline, from feature selection and engineering to target transformations and confidence intervals, is abstracted to a single python Pipeline object that follows the scikit interface for ease of consumption and ease of transparency.
  - A user-centric, two-way [**interpretation interface**][interpretation] that allows for granular interpretation of models and predictions while also allowing domain experts to override factors. (In progress)
  - Abstracted and scalable feature engineering. Computation is handled scalably by the Dask back-end with minimal configuration required by the user and on the cloud provider of the user's choice by leveraging [Dask Cloud Provider](https://cloudprovider.dask.org/en/latest/)
  - Built-in pipeline orchestration tools, such as log collection, task graph synthesis, task parallelization, task automation and artifact tracing leveraging [Prefect](https://www.prefect.io/)
  - Automatic persistence of all experiment artifacts, including models, predictions and validation metrics, to s3 for posterity, traceability and easy integration.
  
  
  [interpretation]: https://github.com/secrettoad/divina
  
## Roadmap
Current development priorities and improvements slated for next and beta release are:

  - Addition of interpretability and interference application that makes consuming, understanding and interacting with forecasts easy and seamless
  - Additional boosting options, such as RNNs, LSTMs, ARIMA, SARIMA, etc.
  - Addition of more realistic test cases, useful error messages and robust documentation
  - Inversion of control of Dask cluster creation, allowing for customization of location and size of cloud compute clusters
   
   

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
docker run jhurdle/divina-storage -p 9000:9000
docker run jhurdle/divina-prefect -p 4200:4200
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

