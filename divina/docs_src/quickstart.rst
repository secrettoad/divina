:noindex:

********************
quickstart
********************

**Date**: |today| **Version**: |version|

Getting Started
###############

At a minimum, a :mod:`divina` pipeline must know which column represents the time index, which column represents the target and the frequency of the time index. Below is a minimal example of a pipeline trained on used to make in-sample predictions on the retail sales dataset from the M5 competition.


.. literalinclude:: _static/pipeline_definitions/quickstart0.py
   :language: python

.. raw:: html
    :file: _static/plots/quickstart/0_s_2_2d.html

As you can see, Divina automatically uses the non-target columns that we don't tell it to drop in the file to make in-sample predictions that are reasonably accurate.

**Hyperparameter Optimization**

Divina supports hyper-parameter optimization through grid search. In order to optimize the parameters of the selected causal model within the pipeline, provide a list of parameter dictionaries to be optimized from.

.. literalinclude:: _static/pipeline_definitions/quickstart1.py
   :language: python

.. raw:: html
    :file: _static/plots/quickstart/1_s_2_2d.html

It seems clear that the log link function improves the performance of the default, linear, causal model on this sales dataset.

However, the forecast produced is for all stores in aggregate while there are three distinct retail locations in the dataset.

**Target Dimensions**

Below we use the target_dimensions option to tell divina to individually aggregate and forecast each retail store in the dataset.

.. literalinclude:: _static/pipeline_definitions/quickstart2.py
   :language: python

.. raw:: html
    :file: _static/plots/quickstart/2_s_1_2d.html

We can see above that the forecast for an individual store is even more accurate and through the interpretability interface below what information is influencing the forecasts and how.

.. raw:: html
    :file: _static/plots/quickstart/2_test_forecast_retail_s_1_factors.html

**Time Features**

An important part of forecasting and feature of divina is the ability to derive time-related features from the time index of a dataset. This is automatically handled by setting the `time_features` attribute of a :mod:`divina`` pipeline to `True`. If only a subset of time features are needed, those that aren't needed can be dropped with the `drop_features` attribute.

.. literalinclude:: _static/pipeline_definitions/quickstart3.py
   :language: python

We can see through the interpretablity interface that the new time information is now informing the forecasts.

.. raw:: html
    :file: _static/plots/quickstart/3_test_forecast_retail_s_1_factors.html

**Feature Engineering**

Information encoding, binning and interaction terms are all powerful features of divina that bring its performance in line with that of tree-based models and neural networks. Here the interpetation interface shows us that our newly engineered features are informing the forecasts.

.. literalinclude:: _static/pipeline_definitions/quickstart4.py
   :language: python

.. raw:: html
    :file: _static/plots/quickstart/4_test_forecast_retail_s_1_factors.html

**Cross Validation**

While visual inspection is a powerful tool for validating a model, programmatic and distributional validation is provided through the ``time_validation_splits`` option of divina.

.. literalinclude:: _static/pipeline_definitions/quickstart5.py
   :language: python

The resulting validation split metrics can be accessed through the `PipelineFitResult` object that is returned from `pipeline.fit()`.

**Confidence Intervals**

Confidence intervals provide important insight into how sure divina is of its predictions, further allowing high-quality decisions to be made on top of them. Below we add confidence intervals to the forecasts via the ``confidence_intervals`` option.

.. literalinclude:: _static/pipeline_definitions/quickstart7.py
   :language: python

.. raw:: html
    :file: _static/plots/quickstart/7_s_1_2d.html


**Endogenous Boosting**

Divina provides the important capability of boosting the residuals of the causal piece of the ensemble, allowing forecasts to be much higher quality and more highly automated. You can see here that the default boosting model, an exponentially weighted moving average, makes small changes in the forecasts using the information available at the specified time horizons.

.. literalinclude:: _static/pipeline_definitions/quickstart8.py
   :language: python

.. raw:: html
    :file: _static/plots/quickstart/8_s_1_2d.html