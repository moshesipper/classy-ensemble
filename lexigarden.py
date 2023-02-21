# Classy Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# class Lexigarden

from basic_ensemble import BasicEnsemble

from sklearn.base import BaseEstimator
from copy import deepcopy
from random import choice, shuffle


def lexicase(models, outputs, targets):
    # Lexicase selection: https://faculty.hampshire.edu/lspector/pubs/lexicase-beyond-gp-preprint.pdf
    assert len(targets) == len(outputs[0]), 'lexicase error: target length does not match output length'
    assert len(models) == len(outputs), 'lexicase error: number of models does not match number of output vectors'
    candidates = list(range(len(models)))
    test_cases = list(range(len(targets)))
    shuffle(test_cases)
    while True:
        case = test_cases[0]
        best_on_first_case = [c for c in candidates if outputs[c][case] == targets[case]]
        if len(best_on_first_case) > 0:
            candidates = best_on_first_case
        if len(candidates) == 1:
            return deepcopy(models[candidates[0]])
        del test_cases[0]
        if len(test_cases) == 0:
            return deepcopy(models[choice(candidates)])


class Lexigarden(BaseEstimator):
    def __init__(self, models, y, garden_size):
        # `models`: list of dicts, with fitted models and other stuff
        m = [model['model'] for model in models]
        o = [model['output'] for model in models]
        lx = [lexicase(models=m, outputs=o, targets=y) for i in range(garden_size)]
        self.ensemble = BasicEnsemble(models=[{'model': model} for model in lx])

    def predict(self, X):
        return self.ensemble.predict(X)

    def size(self):
        return self.ensemble.size()
