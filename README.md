<div align="center">
  <img src="https://storage.googleapis.com/coysuweb-static/assets/images/logo/3.png"><br>
</div>

-----------------

# divina: scalable and hyper-interpretable causal forecasting toolkit
[![Continuous Integration](https://github.com/secrettoad/divina/actions/workflows/prod.yaml/badge.svg)](https://github.com/secrettoad/divina/actions/workflows/prod.yaml)
[![PyPI Latest Release](https://img.shields.io/pypi/v/divina.svg)](https://pypi.org/project/divina/)
[![Package Status](https://img.shields.io/pypi/status/divina.svg)](https://pypi.org/project/divina/)
[![License](https://img.shields.io/pypi/l/divina.svg)](https://github.com/secrettoad/divina/blob/master/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/divina.svg)](https://pypi.org/project/divina)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## What is it?

**divina** is a Python package that provides scalable, interpretable and performant forecasting capabilities designed to make causal forecasting modular, efficient and simple.
It aims to reduce the challenge of causal forecasting on datasets of any size to configuration via JSON as opposed to construction and consumption of Python objects. 
At its core, divina aims reduce the complexity and increase the consistency of causal forecasting at scale.


## Main Features
Here are just a few of the things that divina does well:

  - Abstraction of all necessary configuration of an experiment, from feature selection and engineering to target transformations and confidence intervals, is abstracted to a single JSON file for ease of consumption and ease of transparency.
  - A user-centric, two-way [**interpretation interface**][interpretation] that allows for granular interpretation of models and predictions while also allowing domain experts to override factors.
  - Abstracted and scalable feature engineering. Encoding, interaction, normalization, binning and joining of datasets are handled scalably by the Dask back-end with minimal configuration required by the user.
  - Simulation of user-defined factors in support of forward-looking, multi-signal and decision-enabling causal forecasts.   
  - Automatic persistence of all experiment artifacts, including models, predictions and validation metrics, to s3 for posterity, traceability and easy integration.
  
  
  [interpretation]: https://github.com/secrettoad/divina
  
## Roadmap
Current development priorities and improvements slated for next and beta release are:

  - Addition of automated experiment summaries as persisted artifacts enabling ease of consumption and increased transparency into the forecasts and models divina produces.
  - Improvement of the core model's performance, with the addition of attention mechanisms and the ability to adapt to signals with dynamic mean and variance.  
  - Addition of more realistic test cases, useful error messages and robust documentation.
  - Cleanup of various pieces of the codebase and addition of convenience features such as filepath validation, signal filtering and a maximum lifespan for all EC2 instances divina creates.
   
   

## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/secrettoad/divina

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/divina)

```sh
pip install divina
```

## Dependencies
- [dask - Adds support for arbitrarily large datasets via remote, parallelized compute](https://www.dask.org)
- [dask-ml - Provides distributed-optimized implementations of many popular models](https://ml.dask.org)
- [s3fs - Allows for easy and efficient access to S3](https://github.com/dask/s3fs)
- [pyarrow - Enabled persistence of datasets as storage and compute efficent parquet files](https://arrow.apache.org/docs/python/)



## License
[BSD 3](LICENSE)

## Documentation
Divina' documentation is available [here](https://secrettoad.github.io/divina/#). 

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

