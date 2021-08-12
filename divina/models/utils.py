from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


def compare_sk_models(model1, model2):
    if not isinstance(model1, Pipeline):
        steps1 = [('step1', model1)]
    else:
        steps1 = model1.steps
    if not isinstance(model2, Pipeline):
        steps2 = [('step1', model2)]
    else:
        steps2 = model2.steps
    for s, o in zip(steps1, steps2):
        for i, j in zip(s, o):
            if not type(i) == type(j):
                return False
            if isinstance(i, BaseEstimator):
                if not set(i.get_params()) == set(j.get_params()):
                    return False
    return True
