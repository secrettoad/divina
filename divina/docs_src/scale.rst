:noindex:

**********************
scale
**********************

**Forecasting at Scale**

Inversion of control of the Dask client that connects to, authenticates with and uses for all Divina pipeline computations remote Dask clusters on AWS, GCP, Azure and more via Dask Cloud provider is enabled through the provision of the `dask_configuration` argument to a Divina pipeline's `fit` and `predict` methods.

Below is an example of a pipeline running on AWS.

.. literalinclude:: _static/examples/aws_example.py
   :language: python

