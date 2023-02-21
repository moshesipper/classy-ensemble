# Classy Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# tourchvision datasets

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


img_datasets = []
Datasets = dict()


# MNIST, https://github.com/pytorch/examples/blob/master/mnist/main.py
img_datasets += ['mnist']
Datasets['mnist'] = dict()
Datasets['mnist']['transforms'] = \
    transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
Datasets['mnist']['load'] = \
    lambda root, train: datasets.MNIST(root, train=train, transform=Datasets['mnist']['transforms'])


# Fashion-MNIST, https://pytorch.org/vision/stable/datasets.html#fashion-mnist
img_datasets += ['fashionmnist']
Datasets['fashionmnist'] = dict()
Datasets['fashionmnist']['transforms'] = Datasets['mnist']['transforms']
Datasets['fashionmnist']['load'] = \
    lambda root, train: datasets.FashionMNIST(root, train=train, transform=Datasets['fashionmnist']['transforms'])


# CIFAR10, https://pytorch.org/vision/stable/datasets.html#cifar
img_datasets += ['cifar10']
Datasets['cifar10'] = dict()
Datasets['cifar10']['transforms'] = \
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
Datasets['cifar10']['load'] = \
    lambda root, train: datasets.CIFAR10(root, train=train, transform=Datasets['cifar10']['transforms'])


# CIFAR100 (transforms from https://github.com/shuoros/cifar100-resnet50-pytorch)
img_datasets += ['cifar100']
Datasets['cifar100'] = dict()
Datasets['cifar100']['transforms'] = \
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
Datasets['cifar100']['load'] = \
    lambda root, train: datasets.CIFAR100(root, train=train, transform=Datasets['cifar100']['transforms'])


# ImageNet https://github.com/pytorch/examples/blob/master/imagenet/main.py
img_datasets += ['imagenet']
Datasets['imagenet'] = dict()
Datasets['imagenet']['transforms'] = \
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

Datasets['imagenet']['load'] = \
    lambda root, train: datasets.ImageNet('/cs_storage/public_datasets/ImageNet', split='train' if train else 'val',
                                          transform=Datasets['imagenet']['transforms'])


# Tiny ImageNet, https://www.image-net.org/download.php
img_datasets += ['tinyimagenet']
Datasets['tinyimagenet'] = dict()
Datasets['tinyimagenet']['transforms'] = Datasets['imagenet']['transforms']
Datasets['tinyimagenet']['load'] = \
    lambda root, train: ImageFolder(root + '/tiny-imagenet-200' + ('/train' if train else '/val'),
                                    transform=Datasets['tinyimagenet']['transforms'])


def _dataset_properties(dataset):
    ds = Datasets[dataset]['load'](root='../datasets', train=True)
    loader = DataLoader(ds)
    for data, target in loader:
        break
    return data.shape, len(ds.classes)


for ds in img_datasets:
    Datasets[ds]['input_shape'], Datasets[ds]['n_classes'] = _dataset_properties(ds)
