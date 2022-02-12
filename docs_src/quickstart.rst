:noindex:

********************
quickstart
********************

**Date**: |today| **Version**: |version|

Getting Started
###############

A minimal experiment definition supplies the path to the dataset, the name of the column that holds the time index and the name of the column that holds the target to predict. In the minimal example below, an experiment is conducting using the retail sales data included with divina and using the log link function best suited towards sales data.


.. literalinclude:: _static/experiment_definitions/quickstart1.json
   :language: json

.. raw:: html
    :file: _static/plots/quickstart/quickstart1_h_0_s_6_2d.html

As you can see, Divina automatically uses the non-target data in the file to make insample predictions that are quite accurate. However, the forecast produced is for all stores in aggregate while there are three distinct retail locations in the dataset.

**Target Dimensions**

Below we use the target_dimensions option to tell divina to individually aggregate and forecast each retail store in the dataset.

.. literalinclude:: _static/experiment_definitions/quickstart2.json
   :language: json
   :emphasize-lines: 5-7

.. raw:: html
    :file: _static/plots/quickstart/quickstart2_h_0_s_1_2d.html

We can see through the interpretability interface what information is influencing the forecasts and how.

.. raw:: html
    :file: _static/plots/quickstart/quickstart2_test_forecast_retail_h_0_s_1_factors.html

**Joining Datasets**

An important part of forecasting and feature of divina is the ability to work with additional datasets and the below example definition illustrates how to join the built-in time dataset to the retail dataset, allowing for additional information to be used in the predictions.

.. literalinclude:: _static/experiment_definitions/quickstart3.json
   :language: json
   :emphasize-lines: 10-19

We can see through the interpretablity interface that the new time information is now informing the forecasts. This is important because in order to make long-range forecasts, datasets with forward-looking information or assumptions present through the forecast period must be used.

.. raw:: html
    :file: _static/plots/quickstart/quickstart3_test_forecast_retail_h_0_s_1_factors.html

**Feature Engineering**

Information encoding, binning and interaction terms are all powerful features of divina that bring its performance in line with that of tree-based models and neural networks. Here we narrow the information provided to the model to prevent overfitting and add those options to the experiment definition. You can see that the forecasts become more meaningful through the interpretability interface.

.. literalinclude:: _static/experiment_definitions/quickstart4.json
   :language: json
   :emphasize-lines: 20-49

.. raw:: html
    :file: _static/plots/quickstart/quickstart4_h_0_s_1_2d.html

.. raw:: html
    :file: _static/plots/quickstart/quickstart4_test_forecast_retail_h_0_s_1_factors.html

**Cross Validation**

While visual inspection is a powerful tool for validating a model, programmatic and distributional validation is provided through the ``time_validation_splits`` option of divina.

.. literalinclude:: _static/experiment_definitions/quickstart5.json
   :language: json
   :emphasize-lines: 50-52

**Simulation**

A key feature of divina is the ability to easily simulate potential future values as information to feed the model. In our retail example, we simulate promotions as both occuring and not every day, so that we have both scenarios to consider during the decision-making process.

.. literalinclude:: _static/experiment_definitions/quickstart6.json
   :language: json
   :emphasize-lines: 53-67

.. raw:: html
    :file: _static/plots/quickstart/quickstart6_h_0_s_1_2d.html

.. raw:: html
    :file: _static/plots/quickstart/quickstart6_test_forecast_retail_h_0_s_1_factors.html


**Confidence Intervals**

Confidence intervals provide important insight into how sure divina is of its predictions, further allowing high-quality decisions to be made on top of them. Below we add confidence intervals to the forecasts via the ``confidence_intervals`` option.

.. literalinclude:: _static/experiment_definitions/quickstart7.json
   :language: json
   :emphasize-lines: 68-72

.. raw:: html
    :file: _static/plots/quickstart/quickstart7_h_0_s_1_2d.html