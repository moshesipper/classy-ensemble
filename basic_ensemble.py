# Classy Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# class BasicEnsemble

import numpy as np
from sklearn.base import BaseEstimator


class BasicEnsemble(BaseEstimator):
    def __init__(self, models, topk=None):
        # `models`: list of dicts, with fitted models and other stuff
        if topk is not None:
            srt = sorted(models, key=lambda x: x['score'], reverse=True)
            self.ensemble = [model['model'] for model in srt[:topk]]
        else:
            self.ensemble = [model['model'] for model in models]

    def predict(self, X):
        predictions = np.concatenate([model.predict(X).reshape(-1, 1) for model in self.ensemble], axis=1)
        majority = np.array([np.argmax(np.bincount(row)) for row in predictions])
        return majority

    def size(self):
        return len(self.ensemble)
