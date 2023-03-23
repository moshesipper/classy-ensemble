# Classy Evolutionary Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# class ClassyEnsemblePre
# Classy Ensemble that uses preloaded outputs, used by main_cleen (Classy Evolutionary Ensemble)

import numpy as np
from sklearn.base import BaseEstimator


class ClassyEnsemblePre(BaseEstimator):
    def __init__(self, models, n_classes, topk):
        # `models`: list of dicts
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
                    self.ensemble[model_num] = {'score': srt[k]['score'],
                                                'predictions_train': srt[k]['predictions_train'],
                                                'predictions_test': srt[k]['predictions_test'],
                                                'outputs_train': srt[k]['outputs_train'],
                                                'outputs_test': srt[k]['outputs_test'],
                                                'classes': [0] * n_classes}
                    self.ensemble[model_num]['classes'][c] = 1
                else:
                    self.ensemble[model_num]['classes'][c] = 1

    def predict(self, train=True):
        predictions = None
        for e in self.ensemble:
            sc = self.ensemble[e]['score'] if self.topk > 1 else 1
            cls = self.ensemble[e]['classes']
            outputs = self.ensemble[e]['outputs_train'] if train else self.ensemble[e]['outputs_test']

            p = outputs * sc * cls

            if predictions is None:
                predictions = p
            else:
                predictions += p

        pred = np.array([np.argmax(row) for row in predictions])

        return pred

    def size(self):
        return len(self.ensemble)
