# Classy Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# class ClassyClusterEnsemble

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

################
# from networks import run_ff  # for DL mode
def run_ff(): pass  # remove this when running DL mode
################


class ClassyClusterEnsemble(BaseEstimator):
    def __init__(self, models, n_classes, topk):
        # `models`: list of dicts, with fitted models and other stuff
        self.topk = topk
        # n_models = len(models)

        outputs = [model['output'] for model in models]
        km = KMeans(n_clusters=topk).fit(outputs)
        clusters = km.predict(outputs)
        self.km = km

        clust = {}
        for cluster, model in zip(clusters, models):
            if cluster not in clust.keys():
                clust[cluster] = [model]
            else:
                clust[cluster].append(model)

        self.ensemble = {}
        for u in clust.keys():
            if u not in self.ensemble.keys():
                self.ensemble[u] = {}
            for c in range(n_classes):
                srt = sorted(clust[u], key=lambda x: x['class_scores'][c], reverse=True)
                for k in range(topk):
                    if k < len(srt):
                        model_num = srt[k]['model_num']
                        if model_num not in self.ensemble[u].keys():
                            model = srt[k]['model']
                            score = srt[k]['score']
                            self.ensemble[u][model_num] = {'model': model, 'score': score, 'classes': [0] * n_classes}
                            self.ensemble[u][model_num]['classes'][c] = 1
                        else:
                            self.ensemble[u][model_num]['classes'][c] = 1

    def predict(self, X=None, loader=None, device=None):
        ml = loader is None  # are we using an ensemble of ML models or an ensemble of deep networks
        predictions = None
        for cluster in self.ensemble.keys():
            for mod in self.ensemble[cluster].values():
                clust = self.km.predict(mod.predict(X))
                if clust == cluster:
                    sc = mod['score'] if self.topk > 1 else 1
                    cls = mod['classes']

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
