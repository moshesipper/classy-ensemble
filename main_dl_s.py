# Classy Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# main module for running DL models / load network outputs from saved files

import numpy as np
from sklearn.metrics import accuracy_score

from utils import get_args, rndstr, acc_per_class
from classy_pre import ClassyEnsemblePre
from torch_datasets import Datasets
from generate_outputs import timm_models

# PretrainedModels1 = ['efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']
# PretrainedModels2 = ['vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
# PretrainedModels3 = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
topk = [1, 2, 3, 4, 5]
topk = [1, 2, 3]
timm_models = ['beitv2_base_patch16_224',
               'tf_efficientnet_l2_ns',
               'convnext_large',
               'xcit_large_24_p16_224_dist',
               'swinv2_cr_large_224']
PretrainedModels = timm_models

n_models = len(PretrainedModels)


def load_from_csv(f):
    data = np.loadtxt(f, delimiter=',')
    outputs, predictions, targets = data[:, :-2], data[:, -2].astype(int), data[:, -1].astype(int)
    return outputs, predictions, targets


def main():
    dataset, _, _ = get_args()
    n_classes = Datasets[dataset]['n_classes']
    resfile = f'dl-{dataset}-{rndstr()}.txt'

    with open(resfile, 'w') as f:
        print(f'dataset: {dataset}', file=f)
        print(f'n_classes: {n_classes}', file=f)
        print(f'n_models: {n_models}', file=f)
        print(f'PretrainedModels: {PretrainedModels}', file=f)

    models, targets_train, targets_test = [], None, None
    for i, model_name in enumerate(PretrainedModels):
        outputs_train, predictions_train, targets_train = load_from_csv(f'{dataset}/{model_name}-train.csv')
        outputs_test, predictions_test, targets_test = load_from_csv(f'{dataset}/{model_name}-test.csv')
        models.append({'model': model_name,
                       'model_num': i,
                       'outputs_train': outputs_train,
                       'outputs_test': outputs_test,
                       'predictions_train': predictions_train,
                       'predictions_test': predictions_test,
                       'score': accuracy_score(targets_train, predictions_train),
                       'test_score': accuracy_score(targets_test, predictions_test),
                       'class_scores': acc_per_class(predictions_train, targets_train, n_classes)})

    srt_train = sorted(models, key=lambda d: d['score'], reverse=True)
    srt_test = sorted(models, key=lambda d: d['test_score'], reverse=True)
    with open(resfile, 'a') as f:
        print(f'@, {dataset}, Single model -- best train: {srt_train[0]["score"]:.3f}, {srt_train[0]["model"]} ', file=f)
        print(f'@, {dataset}, Single model -- best test: {srt_test[0]["test_score"]:.3f}, {srt_test[0]["model"]} ', file=f)

    ensemble, ensemble_score, best_k = None, 0, 0
    for k in topk:
        with open(resfile, 'a') as f:
            print(f'creating ClassyEnsemble with topk={k}', file=f)
        ens = ClassyEnsemblePre(models=models, n_classes=n_classes, topk=k)
        pred = ens.predict(train=True)
        acc = accuracy_score(targets_train, pred)
        with open(resfile, 'a') as f:
            print(f'predicting with ClassyEnsemble, topk={k}', file=f)
        if acc > ensemble_score:
            ensemble_score = acc
            ensemble = ens
            best_k = k

    with open(resfile, 'a') as f:
        print(f'predicting with ClassyEnsemble on test set, best_k={best_k}', file=f)
    pred = ensemble.predict(train=False)
    test_acc = accuracy_score(targets_test, pred)

    with open(resfile, 'a') as f:
        print(f'@, {dataset}, ClassyEnsemble -- best test: {test_acc:.3f}, {best_k}', file=f)


##############
if __name__ == "__main__":
    main()
