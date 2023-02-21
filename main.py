# Classy Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# main module for running ML models

from utils import get_args, rndstr, train_val_test_split, load_dataset, acc_per_class
from basic_ensemble import BasicEnsemble
from classy_ensemble import ClassyEnsemble
from lexigarden import Lexigarden
from cluster_ensemble import ClusterEnsemble
from classy_cluster import ClassyClusterEnsemble
from ml_models import Models, ml_kwargs

import numpy as np
import time
from random import choice
from statistics import median
from collections import Counter
from sklearn.metrics import balanced_accuracy_score
from mlxtend.evaluate import permutation_test

EnsembleTypes = [BasicEnsemble, ClassyEnsemble, ClassyClusterEnsemble, Lexigarden, ClusterEnsemble]
time_limit = 36_000  # seconds = 10-hour limit for all replicates
n_models = 250
topk = [1, 2, 3, 5, 20, 50, 100, n_models]


def score(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


def main():
    dataset, resdir, n_replicates = get_args()
    resfile = f'{resdir}/{dataset}_{rndstr()}.txt'
    X, y, n_samples, n_features, n_classes = load_dataset(dataset)

    with open(resfile, 'w') as f:
        print(f'{dataset} (n_samples: {n_samples}, n_features: {n_features}, n_classes: {n_classes})', file=f)
        print(f'n_models: {n_models}', file=f)
        print(f'n_replicates: {n_replicates}', file=f)
        print(f'topk: {topk}', file=f)
        print(f'models: {", ".join([a.__name__ for a in Models])}', file=f)
        print('', file=f)


    # begin replicate runs
    bests = []  # best model per replicate
    scores = dict([(t, []) for t in ['Best'] + EnsembleTypes])  # replicate scores
    sizes = dict([(t, []) for t in EnsembleTypes])  # best ensemble sizes
    ks = dict([(t, []) for t in EnsembleTypes])  # best k values

    tic = time.time()
    for rep in range(n_replicates):
        # training + validation
        models = []
        for i in range(n_models):
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
            alg = choice(Models)
            kwargs = ml_kwargs(alg)
            model = alg(**kwargs).fit(X_train, y_train)  # train model on training set
            y_pred_val = model.predict(X_val)  # predict on validation set
            val_score = score(y_val, y_pred_val)

            models.append({'model': model,
                           'model_num': i,
                           'output': y_pred_val,
                           'score': val_score,
                           'class_scores': acc_per_class(y_val, y_pred_val, n_classes)})

        # best model
        best_model = max(models, key=lambda x: x['score'])  # best model over validation set
        y_pred_test = best_model['model'].predict(X_test)  # test predictions of best model
        test_score = score(y_test, y_pred_test)  # test-set score of best model
        scores['Best'].append(test_score)
        bests.append(best_model['model'].__class__.__qualname__)
        with open(resfile, 'a') as f:
            print(f'rep {rep+1}, {dataset}, Best, {test_score:.3f}, {best_model["model"].__class__.__qualname__}',
                  file=f)

        # ensembles
        for e in EnsembleTypes:
            ensemble, ensemble_score, best_k = None, 0, 0
            for k in topk:
                if e == BasicEnsemble:
                    ens = BasicEnsemble(models=models, topk=k)
                elif e == ClassyEnsemble:
                    ens = ClassyEnsemble(models=models, n_classes=n_classes, topk=k)
                elif e == ClassyClusterEnsemble:
                    ens = ClassyClusterEnsemble(models=models, n_classes=n_classes, topk=k)
                elif e == Lexigarden:
                    ens = Lexigarden(models=models, y=y_val, garden_size=k)
                elif e == ClusterEnsemble:
                    ens = ClusterEnsemble(models=models, n_clusters=k)
                else:
                    exit(f'unknown ensemble type: {e}')

                sc = score(y_val, ens.predict(X=X_val))  # score of ensemble on validation set
                if sc > ensemble_score:  # retain best-k ensemble over validation set
                    ensemble_score = sc
                    ensemble = ens
                    best_k = k

            escore = score(y_test, ensemble.predict(X=X_test))  # test-set score of best-k ensemble
            scores[e].append(escore)
            sizes[e].append(ensemble.size())
            ks[e].append(best_k)
            s = '>' if escore > test_score else '-'
            with open(resfile, 'a') as f:
                print(f'{s}, rep {rep+1}, {dataset}, {e.__name__}, replicate {rep}, {escore:.3f}, '
                      f'{ensemble.size()}, best_k {best_k}',
                      file=f)

        toc = time.time()
        if (toc - tic) > time_limit:
            with open(resfile, 'a') as f:
                print('time limit reached', file=f)
            exit('time limit reached')

    # done replicate runs
    pvals = {}
    significant = {}
    n_sig = 0
    for key, val in scores.items():
        if key != 'Best':
            pvals[key], significant[key] = -1, '-'
            if median(scores[key]) > median(scores['Best']):
                pvals[key] = permutation_test(scores[key], scores['Best'],
                                              method='approximate',
                                              num_rounds=10_000,
                                              func=lambda x, y: np.abs(np.median(x) - np.median(y)))
                if pvals[key] < 0.05:
                    significant[key] = '!'
                    n_sig += 1

    with open(resfile, 'a') as f:
        print(f'', file=f)
        for key in scores.keys():
            if key == 'Best':
                counts = Counter(bests)
                srt_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
                print(f'@, {dataset}, {n_samples}, {n_features}, {n_classes}, {key}, '
                      f'{median(scores[key]):.3f}, {srt_counts}',
                      file=f)
            else:
                unq = "unique" if n_sig == 1 and significant[key] == '!' else "-"
                print(f'@, {dataset}, {n_samples}, {n_features}, {n_classes}, {key.__name__}, '
                      f'{median(scores[key]):.3f}, {median(sizes[key])}, {median(ks[key])}, '
                      f'{pvals[key]:.3f}, {significant[key]}, {unq}',
                      file=f)

        print(f'@ Experiment for dataset {dataset} ended successfully', file=f)


##############
if __name__ == "__main__":
    main()
