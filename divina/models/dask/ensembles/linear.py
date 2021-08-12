from dask_ml.preprocessing import DummyEncoder
from dask_ml.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


class GLASMA(Pipeline):
    def __init__(self, time_index, **kwargs):
        self.time_index = time_index
        self.categorical_features = kwargs.get('categorical_features', None)
        steps = []
        if self.categorical_features:
            steps.append(DummyEncoder(columns=self.categorical_features))
        steps.append(LinearRegression())
        for i, step in enumerate(steps):
            super(GLASMA, self).__init__(steps=[('{}_{}'.format(i, str(step)), step) for step in steps])

    def __repr__(self, N_CHAR_MAX=700):
        return repr('GLASMA: {}'.format(','.join(['{}: {}'.format(key, self.__dict__[key]) for key in self.__dict__])))




