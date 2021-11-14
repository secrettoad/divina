:noindex:

********************
quickstart
********************

**Date**: |today| **Version**: |version|

divina in 5 minutes

In order to perform your first forecast with divina,

The aim of :mod:`divina` is twofold:

1) **to reduce the complexity of configuration for causal forecasting at scale.** This is accomplished by abstracting all configuration to a single JSON file that lets users configure new experiments easily and safely.

2) **to deliver scalable and bidirectionally interpretable causal forcasting models.** These models bring transparency and incremental control to the forecasting process using a variety of coefficient calculation tools, binning and interacting of features and set of link functions that enable a linear model to fit various target distributions.

**Experiment Definition**


**Experiment Persistence**

Experiment artifacts are persisted either locally or to S3 depending on the use of the `--local` flag when running the experiment command and will produce a local output structure as shown below::

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
