# Classy Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# class ClassyEnsemble

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


################
# from networks import run_ff  # for DL mode
def run_ff(): pass  # remove this when running DL mode
################


class ClassyEnsemble(BaseEstimator):
    def __init__(self, models, n_classes, topk):
        # `models`: list of dicts, with fitted models and other stuff
        self.topk = topk
        n_models = len(models)
        self.ensemble = {}
        for c in range(n_classes):
            if topk < n_models:
                srt = sorted(models, key=lambda x: x['class_scores'][c], reverse=True)
            else:
                srt = models
            for k in range(topk):
                model_num = srt[k]['model_num']
                if model_num not in self.ensemble.keys():
                    model = srt[k]['model']
                    score = srt[k]['score']
                    self.ensemble[model_num] = {'model': model, 'score': score, 'classes': [0] * n_classes}
                    self.ensemble[model_num]['classes'][c] = 1
                else:
                    self.ensemble[model_num]['classes'][c] = 1

    def predict(self, X=None, loader=None, device=None):
        ml = loader is None  # are we using an ensemble of ML models or an ensemble of deep networks
        predictions = None
        for e in self.ensemble:
            mod = self.ensemble[e]['model']
            sc = self.ensemble[e]['score'] if self.topk > 1 else 1
            cls = self.ensemble[e]['classes']

            if ml:  # ML
                p = mod.predict_proba(X) * sc * cls
            else:  # DL
                _, y, _, outputs = run_ff(mod, loader, device)
                p = outputs * sc * cls

            if predictions is None:
                predictions = p
            else:
                predictions += p

        pred = np.array([np.argmax(row) for row in predictions])

        if ml:  # ML
            return pred
        else:  # DL
            acc = accuracy_score(y, pred)
            return pred, acc

    def size(self):
        return len(self.ensemble)
