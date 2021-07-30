from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


class DivinaSparkModel:
    def __init__(self):
        pass

    def __eq__(self, other):
        for s, o in zip(self.stages, other.stages):
            if not s.get_params() == o.get_params():
                return False
        return True


class DivinaDaskModel(Pipeline):
    def __init__(self, steps):
        super(DivinaDaskModel, self).__init__(steps=steps)
        pass

    def __eq__(self, other):
        for s, o in zip(self.steps, other.steps):
            for i, j in zip(s, o):
                if not type(i) == type(j):
                    return False
                if isinstance(i, BaseEstimator):
                    if not set(i.get_params()) == set(j.get_params()):
                        return False
        return True
