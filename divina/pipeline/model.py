from abc import ABC
from functools import partial
import numpy as np
from dask_ml.linear_model import LinearRegression
from sklearn.base import BaseEstimator
import warnings
import dask.array as da


class EWMA(BaseEstimator):
    def __init__(self, alpha: float=0.8):
        self.alpha = alpha
        self._window = None
        self.weights = None
        super().__init__()

    def fit(self, X, y):
        self.window = X.shape[1]

    def predict(self, X):
        if X.shape[1] != self.window:
            raise ValueError('Dataframe passed must have the same number of columns as the data used to train')
        y_hat = da.dot(da.nan_to_num(X), da.from_array(self.weights)) / self.window
        return y_hat

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, value):
        self._window = value
        self.weights = list([(1 - self.alpha) ** n for n in range(self._window)])
        return

    def __eq__(self, other):
        return self.alpha == other.alpha and self.window == other.window and self.weights == self.weights




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

    @property
    def _coef(self):
        return self.linear_model._coef

    @_coef.setter
    def _coef(self, value: np.array):
        self.linear_model._coef = value

    def fit(self, X, y):
        if self.link_function == 'log':
            y = np.log(y)
        self.linear_model.fit(X, y)

    def predict(self, X):
        y_hat = self.linear_model.predict(X)
        if self.link_function == 'log':
            y_hat = np.exp(y_hat)
        return y_hat

    def __eq__(self, other):
        return np.allclose(self._coef, other._coef)




