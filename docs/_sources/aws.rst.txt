:noindex:

**********************
aws
**********************

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
