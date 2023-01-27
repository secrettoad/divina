.. divina documentation master file, created by

.. module:: divina

********************
divina
********************

**Date**: |today|

**Useful links**:
`Binary Installers <https://pypi.org/project/divina>`__ |
`Source Repository <https://github.com/secrettoad/divina>`__ |
`Issues & Ideas <https://github.com/secrettoad/divina/issues>`__ |
`Q&A Support <https://stackoverflow.com/questions/tagged/divina>`__

.. toctree::
   :maxdepth: 1

   quickstart
   orchestration
   scale

:mod:`divina` is an open source, BSD3-licensed library providing scalable and hyper-interpretable causal forecasting capabilities written in `Python <https://www.python.org/>`__ and consumable via CLI.

The aim of :mod:`divina` is to deliver performance-oriented and hypter-interpretable exogenous time series forecasting models by producing accurate and bootstrapped predictions, local and overridable factor summaries and easily configurable feature engineering and experiment management capabilities.

Ensemble Architecture
***********************

:mod:`divina` is essentially a convenience wrapper that facilitates training, prediction, validation and deployment of an ensemble consisting of a causal, interpretable model that is boosted by an endogenous time-series model, allowing for high levels of automation and accuracy while still emphasizing and relying on the causal relationships discovered by the user. This ensemble structure is delivered with swappable model types to be able to suit many different kinds of forecasting problems. :mod:`divina` is also fully integrated with both Dask and Prefect meaning that distributed compute and pipeline orchestration can be enabled with the flip of a switch. For more information of :mod:`divina`'s features, check out the :doc:`quickstart` page.

Installation
************

:mod:`divina` is available via pypi and can be installed using the python package manager pip as shown below.

.. code-block:: bash

    pip install divina


Getting Started
************************

To train and predict using a :mod:`divina` pipeline, we first create a :mod:`pandas` dataframe full of dummy data, convert that to a dask dataframe, and call the `fit()` method of our pipeline. Once the pipeline is fit, it can be used to predict on out-of-sample feature sets.

.. literalinclude:: ../docs_heavy_assets/examples/base.py
   :language: python