.. divina documentation master file, created by

.. module:: divina

********************
divina
********************

**Date**: |today| **Version**: |version|

**Useful links**:
`Binary Installers <https://pypi.org/project/divina>`__ |
`Source Repository <https://github.com/secrettoad/divina>`__ |
`Issues & Ideas <https://github.com/secrettoad/divina/issues>`__ |
`Q&A Support <https://stackoverflow.com/questions/tagged/divina>`__

.. toctree::
   :maxdepth: 1

   cli
   quickstart

:mod:`divina` is an open source, BSD3-licensed library providing scalable and hyper-interpretable causal forecasting capabilities written in `Python <https://www.python.org/>`__ and consumable via CLI.

The aim of :mod:`divina` is to deliver performance-oriented and hypter-interpretable exogenous time series forecasting models by producing accurate and bootstrapped predictions, local and overridable factor summaries and easily configurable feature engineering and experiment management capabilities.

Installation
************

:mod:`divina` is available via pypi and can be install using the python package manager pip as shown below.

.. code-block:: bash

    pip install divina

Use
************

:mod:`divina` is consumable via CLI, or command line interface. In order to run an experiment with :mod:`divina`, first create your experiment definition and then run the below command in your console of choice.

.. code-block:: bash

    divina experiment /path/to/my/experiment_definition.json

Experiment Definitions
************

Experiment configuration with :mod:`divina` has been abstracted completely to a JSON file called the experiment definition that the user supplies to the :mod:`divina` cli. For an exhaustive example of an experiment definition with every available option described in detail, check out the experiment definition documentation. Or if you are new to divina, check out the get started page.

Getting Started
************

To run an experiment with divina, first install it and then create an experiment definition that describes your experiment. Here we create a minimal experiment definition that allows us to run a forecasting experiment using the retail sales and time data included with divina.

.. code-block:: json

    {
      "experiment_definition": {
        "target": "Sales",
        "time_index": "Date",
        "data_path": "divina://retail_sales"
      }
    }





.. code-block:: json

    {
      "experiment_definition": {
        "target": "<string>",
        "signal_dimensions": ["<string>"],
        "link_function": "<string>",
        "time_horizons": ["<integer>"],
        "time_index": "<string>",
        "include_features": ["<string>"],
        "drop_features": ["<string>"],
        "time_validation_splits": ["<string>"],
        "train_end": "<string>",
        "train_end": "<string>",
        "forecast_end": "<string>",
        "forecast_end": "<string>",
        "validation_end": "<string>",
        "validation_end": "<string>",
        "encode_features": ["<string>"],
        "scenarios": [
            {
              "feature": "<string>",
              "values": ["<string, integer, float>"],
              "start": "<string>",
              "end": "<string>"
            }
        ],
        "scenario_freq": "<string>",
        "data_path": "<string>",
        "confidence_intervals": ["<integer>", "<integer>"],
        "joins": [
          {
            "data_path": "<string>",
            "join_on": ["<string>", "<string>"],
            "as": "<string>"
          }
        ]
        "bootstrap_sample": "<integer>",
      }
    }



.. code-block:: json

    {
        "experiment_definition": {
            "time_index": "Date",
            "target": "Sales",
            "include_features": ["Store", "Promo", "Weekday",
                                 "LastDayOfMonth"],
            "time_validation_splits": ["2015-07-18"],
            "forecast_end": "2015-08-30",
            "bootstrap_sample": 5,
            "signal_dimensions": ["Store"],
            "time_horizons": [2],
            "forecast_freq": "D",
            "encode_features": ["Weekday", "Store"],
            "scenarios": [{"feature": "Promo", "values": [0, 1], "start": "2015-08-01", "end": "2016-01-01"}],
            "dataset_directory": "divina://retail_sales",
            "link_function": "log",
            "confidence_intervals": [100, 0],
            "joins": [
                {
                    "dataset_directory": "divina://time",
                    "join_on": ["Date", "Date"],
                    "as": "time"
                }
            ]
        }
    }

**Experiment Persistence**

Experiment artifacts are persisted either locally or to S3 depending on the use of the `--aws` flag as structured below.

    experiment path
      |- models
      |    |
      |    \- h_{forecast horizon}
      |           |-fit_model.joblib
      |           |-bootstrap
      |                |
      |                |- bootstrap_model_{random seed}
      |
      |- forecast
      |    |
      |    |- common_meta.parquet
      |    |- forecast_partition_0_meta.parquet
      |    |- forecast_partition_0.parquet
      |    \  ...
      |
      |- validation
           |
           |- metrics.json
           \- {validation split}
                  |
                  |- validation_partition_0_meta.parquet
                  |- validation_partition_0.parquet
                  \  ...
