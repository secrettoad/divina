:noindex:

**********************
orchestration
**********************

**Pipeline Orchestration with Prefect**

Divina pipelines have built-in orchestration via Prefect, meaning that by only setting a keyword argument to `True`, fitting and predicting with your Divina pipelines store, track and visualize each incremental task and artifact passed between tasks. Combined with the easy configuration of the underlying Dask cluster, this means your pipelines can be taken into production extremely easily.

Below is an example of fitting and predicting with a Divina pipeline using the built-in Prefect orchestration (Please be sure to use the appropriate environment variables to point and authenticate to your Prefect service).

.. literalinclude:: _static/examples/prefect_example.py
   :language: python
   :emphasize-lines: 10-19

**Artifact Persistence**

In order to persist pipeline artifacts (datasets, models and metrics), one must only set the `test_pipeline_root` attribute of their Divina pipeline.

Below is an example of artifact persistence to a local path:

.. literalinclude:: _static/examples/local_artifact.py
   :language: python
   :emphasize-lines: 10-19

Below is an example of artifact persistence to S3 (be sure to set the appropriate credentials via environment variable):

.. literalinclude:: _static/examples/s3_artifact.py
   :language: python
   :emphasize-lines: 10-19