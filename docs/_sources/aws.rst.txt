:noindex:

**********************
aws
**********************

**Forecasting at Scale**

In order to to work with larger datasets, include more features, increase the bootstrap sample of divina's confidence intervals, or otherwise scale your forecasting workload, use the --aws_workers option when running the experiment through the cli.

.. code-block:: bash

    divina experiment /path/to/my/experiment_definition.json -aws_workers=10