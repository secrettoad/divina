:notoc:

.. divina documentation master file, created by

.. module:: divina

********************
divina documentation
********************

**Date**: |today| **Version**: |version|

**Useful links**:
`Binary Installers <https://pypi.org/project/divina>`__ |
`Source Repository <https://github.com/secrettoad/divina>`__ |
`Issues & Ideas <https://github.com/secrettoad/divina/issues>`__ |
`Q&A Support <https://stackoverflow.com/questions/tagged/divina>`__ |
`Mailing List <mailto:partners@coysu.com>`__

:mod:`divina` is an open source, BSD3-licensed library providing scalable and hyper-interpretable causal forecasting capabilities written in `Python <https://www.python.org/>`__ and consumable either via Bash CLI or the built-in web-app.
programming language.

The aim of :mod:`divina` is twofold:

1) to reduce the complexity of configuration for causal forecasting at scale. this is accomplished by abstracting all configuration to a single JSON file that lets users configure new experiments easily and safely. Below is an example forecast definition.

.. code-block:: json

    {
        "vision_definition": {
            "time_index": "index",
            "target": "passengers",
            "time_validation_splits": ["1957-01-01"],
            "time_horizons": [1],
            "dataset_directory": "s3://divina-public/dataset",
            "dataset_id": "airline_sales"
        }
    }

2) to deliver scalable and bidirectionally interpretable models that bring transparency and incremental control to the forecasting process. This is done using a variety of coefficient calculation tools for highly-parametric and non-parametric models, binning and interacting of features and and a set of interfaces allowing users to override individual model and forecast coefficients with domain knowledge.

In a minimal example, divina can be used to create a weather forecast using the below command and forecast definition

.. code-block:: bash

    divina forecast forecast_definition.json --local


and will produce a local output structure as shown below::

    divina-forecast
      |- models
      |    |
      |    \- insample
      |         |
      |         \- model.joblib
      |- predictions
      |      |
      |      \- insample
      |           |
      |           \- predictions_partition_0.parquet
      \- validation
             |
             \- insample
                  |
                  \- metrics.json

In a more advanced configuration, divina can be used with the following command and forecast definition

to produce an s3 hosted output structure as shown below

###TODO visualization and interpretation interface

.. click:: divina.cli.cli:divina
   :prog: divina
   :nested: full


