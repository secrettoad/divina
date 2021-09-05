<div align="center">
  <img src="https://storage.googleapis.com/coysuweb-static/assets/images/logo/divina_logo.png"><br>
</div>

-----------------

# divina: scalable and hyper-interpretable causal forecasting toolkit
[![Continuous Integration](https://github.com/secrettoad/divina/actions/workflows/prod.yaml/badge.svg)](https://github.com/secrettoad/divina/actions/workflows/prod.yaml)
[![PyPI Latest Release](https://img.shields.io/pypi/v/divina.svg)](https://pypi.org/project/divina/)
[![Package Status](https://img.shields.io/pypi/status/divina.svg)](https://pypi.org/project/divina/)
[![License](https://img.shields.io/pypi/l/divina.svg)](https://github.com/pandas-dev/divina/blob/master/LICENSE)
[![Coverage](https://codecov.io/github/secrettoad/divina/coverage.svg?branch=main)](https://codecov.io/gh/secrettoad/divina)
[![Downloads](https://img.shields.io/pypi/dm/divina.svg)](https://pypi.org/project/divina)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## What is it?

**divina** is a Python package that provides scalable, interpretable and accurate forecasting capabilities designed to make causal forecasting modular, efficient and simple.
It aims to be reduce the challenge of causal forecasting on datasets of any size to experiment configuration via JSON and, if the extensive set of pre-implemented models is insufficient, custom ensemble design via scikit-learn.
More specifically, divina aims to raise the standard of functionality in forecasting by implementing two-way interpretability interfaces for all models.


## Main Features
Here are just a few of the things that pandas does well:

  - Easy construction of parquet-based [**datasets**][datasets] that allow for efficient, scalable modelling and dynamic, JSON-driven definition of multi-dataset experiments.
  - Abstraction of granular experiment configuration such as cross-validation, regularization and metric selection to a single, simple JSON-based configuration file with sensible defaults.  
  - Automatic persistence of all experiment artifacts, including models, predictions and validation metrics, to s3 for posterity, traceability and easy integration.
  - A user-centric, two-way [**interpretation interface**][interpretation] that allows for granular interpretation of models and predictions while also allowing domain experts to override factors.


   [datasets]: https://github.com/secrettoad/divina
   [interpretation]: https://github.com/secrettoad/divina
   

## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/secrettoad/divina

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/divina)

```sh
pip install pandas
```

## Dependencies
- [dask - Adds support for arbitrarily large datasets via remote, parallelized compute](https://www.dask.org)
- [dask-ml - Provides distributed-optimized implementations of many popular models](https://ml.dask.org)
- [s3fs - Allows for easy and efficient access to S3](https://github.com/dask/s3fs)
- [pyarrow - Enabled persistence of datasets as storage and compute efficent parquet files](https://arrow.apache.org/docs/python/)



## License
[BSD 3](LICENSE)

## Documentation
Robust documentation is currently in the works.

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

