:noindex:

**********************
experiments
**********************

**Experiment Definitions**

Below is the schema for an experiment, illustrating all of the different options and their expected data types.

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
