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

   quickstart
   experiments
   aws
   cli

:mod:`divina` is an open source, BSD3-licensed library providing scalable and hyper-interpretable causal forecasting capabilities written in `Python <https://www.python.org/>`__ and consumable via CLI.

The aim of :mod:`divina` is to deliver performance-oriented and hypter-interpretable exogenous time series forecasting models by producing accurate and bootstrapped predictions, local and overridable factor summaries and easily configurable feature engineering and experiment management capabilities.


Installation
************

:mod:`divina` is available via pypi and can be install using the python package manager pip as shown below.

.. code-block:: bash

    pip install divina


Getting Started
************************

To run an experiment with divina, first install it and then create an experiment definition that describes your experiment. Here we create a minimal experiment definition that allows us to run a forecasting experiment using the retail sales and time data included with divina.

.. code-block:: json

    {
      "experiment_definition": {
        "target": "Sales",
        "time_index": "Date",
        "data_path": "divina://retail_sales"
      }
    }


Use
************

:mod:`divina` is consumable via CLI, or command line interface. In order to run an experiment with :mod:`divina`, first create your experiment definition and then run the below command in your console of choice.

.. code-block:: bash

    divina experiment /path/to/my/experiment_definition.json