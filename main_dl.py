# Classy Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# main module for running DL models

from utils import get_args, rndstr, acc_per_class
from classy_ensemble import ClassyEnsemble
from networks import PretrainedModels, run_ff
from torch_datasets import Datasets

import torch
import torchvision.models as vismodels


# PretrainedModels = ['efficientnet_b7', 'regnet_y_32gf', 'resnext101_32x8d', 'wide_resnet101_2', 'resnet101']
# PretrainedModels = ['efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']
# PretrainedModels = ['resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50']
# PretrainedModels = ['vgg11', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
PretrainedModels = ['regnet_y_32gf', 'resnext101_32x8d', 'wide_resnet101_2', 'resnet101', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
n_models = len(PretrainedModels)
# topk = [1, 2, 3, 5, 11]  # , 20, n_models]

topk = [2, 3]


def main():
    # torch.set_default_tensor_type(torch.float16)
    dataset, resdir, _ = get_args()
    resfile = f'{resdir}/{dataset}_{rndstr()}.txt'

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    input_shape, n_classes = Datasets[dataset]["input_shape"], Datasets[dataset]["n_classes"]

    with open(resfile, 'w') as f:
        print(f'{dataset} ({input_shape}, {n_classes})', file=f)
        print(f'n_models: {n_models}', file=f)
        print(f'topk: {topk}', file=f)
        print(f'PretrainedModels: {PretrainedModels}', file=f)

    train_set = Datasets[dataset]['load'](root='../datasets', train=True)
    train_set_size = int(len(train_set) * 0.75)
    validation_set_size = len(train_set) - train_set_size
    train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, validation_set_size])

    test_set = Datasets[dataset]['load'](root='../datasets', train=False)

    with open(resfile, 'a') as f:
        print(f'{dataset} lens train/validation/test sets: {len(train_set)}/{len(validation_set)}/{len(test_set)}\n',
              file=f)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64)

    # training + validation
    best_test, best_model = 0, None
    models = []
    for i, model_name in enumerate(PretrainedModels):
        if dataset == 'imagenet':
            model = vismodels.__dict__[model_name](pretrained=True).to(device)
        else:  # cifar10, cifar100, fashionmnist, tinyimagenet
            model = torch.load(f'../deepgold/Area51/{dataset}/{model_name}.pt')

        # train_score, y, y_pred, _ = run_ff(model, train_loader, device)
        validation_score, y, y_pred, _ = run_ff(model, validation_loader, device)

        models.append({'model': model,
                       'model_num': i,
                       'output': None,  # not used in DL mode
                       #  'score': train_score,
                       'score': validation_score,
                       'class_scores': acc_per_class(y, y_pred, n_classes)})

        test_score, _, _, _ = run_ff(model, test_loader, device)
        if test_score > best_test:
            best_test = test_score
            best_model = model_name

        with open(resfile, 'a') as f:
            # print(f'({i+1}) {model_name}, train_score: {train_score:.3f}, test_score: {test_score:.3f}, '
            print(f'({i+1}) {model_name}, validation_score: {validation_score:.3f}, test_score: {test_score:.3f}, '
                  f'best so far: {best_test:.3f}, {best_model}', file=f)

    ensemble, ensemble_score, best_k = None, 0, 0
    for k in topk:
        ens = ClassyEnsemble(models=models, n_classes=n_classes, topk=k)
        _, acc = ens.predict(X=None, loader=train_loader, device=device)
        if acc > ensemble_score:
            ensemble_score = acc
            ensemble = ens
            best_k = k

    _, acc = ensemble.predict(loader=test_loader, device=device)

    with open(resfile, 'a') as f:
        print(f'@, Best test, {dataset}, {best_test:.3f}, {best_model}', file=f)
        print(f'@, ClassyEnsemble, {dataset}, {acc:.3f}, {best_k}', file=f)


##############
if __name__ == "__main__":
    main()
