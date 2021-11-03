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

:mod:`divina` is an open source, BSD3-licensed library providing scalable and hyper-interpretable causal forecasting capabilities written in `Python <https://www.python.org/>`__ and consumable via Bash CLI.
programming language.

The aim of :mod:`divina` is twofold:

1) to reduce the complexity of configuration for causal forecasting at scale. This is accomplished by abstracting all configuration to a single JSON file that lets users configure new experiments easily and safely.

2) to deliver scalable and bidirectionally interpretable models that bring transparency and incremental control to the forecasting process. This is done using a variety of coefficient calculation tools, binning and interacting of features and set of link functions that enable a linear model to fit various target distributions.


Experiment Persistence

Experiment artifacts are persisted either locally or to S3 depending on the use of the `--local` flag when running the experiment command.

and will produce a local output structure as shown below::

    experiment path
      |- models
      |    |
      |    |- h_{forecast horizon}
      |    \  ...
      |
      |- forecast
      |    |
      |    |- common_meta.parquet
      |    |- forecast_partition_0_meta.parquet
      |    |- forecast_partition_0.parquet
      |    \  ...
      |
      |- validation
      |      |
      |      \- {validation split}
      |            |- validation_partition_0_meta.parquet
      |            |- validation_partition_0.parquet
      |            \  ...
      |
      \- metrics.json

.. click:: divina.cli.cli:divina
   :prog: divina
   :nested: full


