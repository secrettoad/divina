from abc import ABC
from functools import partial
import numpy as np
from dask_ml.linear_model import LinearRegression
from sklearn.base import BaseEstimator
import warnings
import dask.array as da


class EWMA(BaseEstimator):
    def __init__(self, alpha: float, window: int):
        self.alpha = alpha
        self.window = window
        self.weights = list([(1 - self.alpha) ** n for n in range(self.window)])
        super().__init__()

    def fit(self, X, y):
        warnings.warn('There is no need to fit EWMModel. The weights are determined solely by the alpha and window parameters')

    def predict(self, X):
        if X.shape[1] != self.window:
            raise ValueError('Dataframe passed must have the same number of columns as the model\'s window attribute')
        y_hat = da.dot(da.nan_to_num(X), da.from_array(self.weights)) / self.window
        return y_hat


class GLM(BaseEstimator):
    def __init__(self, link_function=None, linear_parameters: dict=None):
        self.link_function = link_function
        self.linear_parameters = linear_parameters
        self.linear_model = None
        if self.linear_parameters:
            self.linear_model = LinearRegression(**linear_parameters)
        else:
            self.linear_model = LinearRegression()
        super().__init__()

    def fit(self, X, y):
        if self.link_function == 'log':
            y = np.log(y)
        self.linear_model.fit(X, y)

    def predict(self, X):
        y_hat = self.linear_model.predict(X)
        if self.link_function == 'log':
            y_hat = np.exp(y_hat)
        return y_hat


