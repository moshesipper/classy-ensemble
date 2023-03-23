# Classy Evolutionary Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# generate output vectors for pretrained models

import argparse
import numpy as np
import torch
import torchvision.models as vismodels
from torch_datasets import Datasets
from networks import PretrainedModels
import timm

# https://github.com/pprp/timm
# https://github.com/pprp/timm/blob/master/results/results-imagenet.csv
# timm.list_models()
timm_models = ['maxvit_xlarge_224',
               'beit_base_patch16_224',
               'beitv2_base_patch16_224',
               'tf_efficientnet_l2_ns',
               'tf_efficientnet_l2_ns_475',
               'volo_d5_224',
               'convnext_large',
               'swin_large_patch4_window7_224',
               'xcit_large_24_p16_224_dist',
               'swinv2_cr_large_224',
               'vit_huge_patch14_224',
               'deit3_huge_patch14_224']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', dest='dataset', type=str, action='store', default='cifar10',
                        help='Dataset to use (default: cifar10)')
    args = parser.parse_args()
    return args.dataset


def main():
    dataset = get_args()
    assert dataset in ['fashionmnist', 'cifar10', 'cifar100', 'tinyimagenet', 'imagenet'],\
        f'unknown dataset {dataset}'
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'batch_size': 64}
    if use_cuda:
        kwargs.update({'num_workers': 1, 'pin_memory': False, 'shuffle': False})

    ds = dict()
    ds['train'] = Datasets[dataset]['load'](root='../datasets', train=True)
    ds['test'] = Datasets[dataset]['load'](root='../datasets', train=False)

    if dataset == 'imagenet':
        t1 = int(len(ds['train']) * 0.9)
        t2 = len(ds['train']) - t1
        _, ds['train'] = torch.utils.data.random_split(ds['train'], [t1, t2])  # keep only t2 samples


    for model_name in PretrainedModels + timm_models:
        if dataset == 'imagenet':
            if model_name in PretrainedModels:
                model = vismodels.__dict__[model_name](pretrained=True).to(device)
            else:  # timm_models
                model = timm.create_model(model_name, pretrained=True).to(device)
        else:
            model = torch.load(f'../deepgold/Area51/{dataset}/{model_name}.pt')

        model.eval()

        for phase in ['train', 'test']:
            if use_cuda:
                torch.cuda.empty_cache()

            loader = torch.utils.data.DataLoader(ds[phase], **kwargs)

            outputs, predictions, targets = None, None, None
            with torch.no_grad():
                for inpt, target in loader:
                    inpt, target = inpt.to(device), target.to(device)

                    target = target.cpu().numpy()
                    targets = target if targets is None else np.concatenate((targets, target), axis=0)

                    output = model(inpt)
                    preds = output.argmax(dim=1, keepdim=True).cpu().numpy()
                    predictions = preds if predictions is None else np.concatenate((predictions, preds), axis=0)

                    output = output.cpu().numpy()
                    outputs = output if outputs is None else np.concatenate((outputs, output), axis=0)

            predictions = predictions.reshape(-1, 1)
            targets = targets.reshape(-1, 1)
            arr = np.concatenate((outputs, predictions, targets), axis=1)
            print(f'Saving {model_name}, arr.shape: {arr.shape}')
            filename = f'{model_name}-{phase}.csv'
            fmt = '%f,' * (outputs.shape[1]) + '%d, %d'
            np.savetxt(f'{dataset}/{filename}', arr, delimiter=',', fmt=fmt)


if __name__ == '__main__':
    main()
