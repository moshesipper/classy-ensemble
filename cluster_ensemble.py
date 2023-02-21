# Classy Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# class ClusterEnsemble

from basic_ensemble import BasicEnsemble

from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans


def cluster_based_pruning(models, n_clusters):
    # construct ensemble using cluster-based pruning
    # `models`: list of dicts
    outputs = [model['output'] for model in models]
    km = KMeans(n_clusters=n_clusters).fit(outputs)
    # n_clusters = len(km.cluster_centers_)  # note:  Number of distinct clusters may end up being < n_clusters
    clusters = km.predict(outputs)

    ensemble = {}
    for cluster, model in zip(clusters, models):
        if cluster not in ensemble.keys():
            ensemble[cluster] = {'model': model['model'], 'score': model['score']}
        elif model['score'] > ensemble[cluster]['score']:
            ensemble[cluster] = {'model': model['model'], 'score': model['score']}

    return BasicEnsemble(models=[{'model': v['model']} for v in ensemble.values()])


class ClusterEnsemble(BaseEstimator):
    def __init__(self, models, n_clusters):
        # `models`: list of dicts, with fitted models and other stuff
        self.ensemble = cluster_based_pruning(models=models, n_clusters=n_clusters)

    def predict(self, X):
        return self.ensemble.predict(X)

    def size(self):
        return self.ensemble.size()
