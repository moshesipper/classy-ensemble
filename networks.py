# Classy Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# network models

import torch.nn as nn
import numpy as np
import torch
from torch_datasets import Datasets
import torchvision.models as vismodels
import gc


# PretrainedList = get_pretrained_list() # run once and save
PretrainedModels = ['alexnet', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'mnasnet0_5', 'mnasnet1_0', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'vgg11', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet101_2']


def get_pretrained_list():
    names = sorted(name for name in vismodels.__dict__ if name.islower() and not name.startswith("__") and callable(vismodels.__dict__[name]))     
    further_treatment = ['inception_v3', 'mnasnet0_75', 'mnasnet1_3', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
                         'squeezenet1_0', 'squeezenet1_1', 'googlenet']
    for model in further_treatment:  # These models require additional processing or are not supported
        names.remove(model)
    return names


class Net(nn.Module):
    def __init__(self, net, dataset, input_shape=None):
        assert net in PretrainedModels, f'Error: Given {net}, must by one of: {PretrainedModels}.'
        super(Net, self).__init__()
        
        model = vismodels.__dict__[net](pretrained=True)
        model = self.add_dropout(model)
        
        n_classes = Datasets[dataset]['n_classes']
        if input_shape is None:
            input_shape = Datasets[dataset]['input_shape']        
        
        self.modify_network(model, net, input_shape, n_classes)
        self.main = model


    def forward(self, x):
        return self.main(x)
    

    def add_dropout(self, model):
        # add dropout layer after each conv
        for name, module in model._modules.items():
            if len(list(module.children())) > 0: # recurse
                self.add_dropout(module)
            else:
                if isinstance(module,nn.modules.conv.Conv2d):
                    model._modules[name] = nn.Sequential(*[model._modules[name] , nn.Dropout2d(0.03)])
        return model


    def modify_network(self, model, net, input_shape, n_classes, fc_type='multiple'):
    # replace first and last layers
        first = self.first_conv_layer(net, in_channels=input_shape[1])
        self.replace_layer(model, first, n_classes, fc_type, done=[False], first=True) # replace first layer        
        # remove_grad(model) # after first, before last, so that grads of last layer remain        
        self.replace_layer(model, None, n_classes, fc_type, done=[False], first=False) # replace last layer


    def first_conv_layer(self, net, in_channels):
        if net in ['mnasnet0_5', 'mobilenet_v3_large', 'mobilenet_v3_small']:
            out_channels = 16
        elif net in ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0']:
            out_channels = 24
        elif net in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'inception_v3', 'mnasnet1_0',\
                     'mobilenet_v2', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf',\
                     'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_16gf', 'regnet_y_1_6gf',\
                     'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf']:
            out_channels = 32
        elif net in ['efficientnet_b3']:
            out_channels = 40
        elif net in ['efficientnet_b4', 'efficientnet_b5']:
            out_channels = 48
        elif net in ['efficientnet_b6']:
            out_channels = 56
        elif net in ['densenet161']:
            out_channels = 96
        else: # default
            out_channels = 64

        if 'vgg' in net or net=='inception_v3':
            layer = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        else:
            layer = nn.Conv2d(in_channels, out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        return layer
    
    
    def replace_layer(self, model, new, n_classes, fc_type, done, first=True):
        '''    
        replace either first or last layer in `model' to `new'
        assume first layer is Conv2d and last layer is Linear
        done is a size-1 list to induce call by reference in recursive calls
        based on: https://discuss.pytorch.org/t/how-to-modify-a-pretrained-model/60509/13
            also: https://www.kaggle.com/ankursingh12/why-use-setattr-to-replace-pytorch-layers
            also: https://discuss.pytorch.org/t/how-can-i-replace-an-intermediate-layer-in-a-pre-trained-network/3586/7  
        '''
        if first:  
            iter = model._modules.items()   # model.named_children()
        else:
            iter = reversed(model._modules.items())
        for name, module in iter:
            if len(list(module.children())) > 0: # recurse
                self.replace_layer(module, new, n_classes, fc_type, done, first)
            elif not done[0]:
                if first:
                    model._modules[name] = new
                else:
                    model._modules[name] = self.fc(model._modules[name].in_features, n_classes, fc_type) 
                done[0] = True
                return
            else:
                return

    
    def fc(self, num_ftrs , n_classes, fc_type='multiple'):
        if fc_type == 'multiple':
            inp = num_ftrs
            out = self.closest_power_of_2(inp)
            layers= []
            while out > n_classes:
                layers += [nn.Linear(inp, out),
                           nn.Dropout(0.05),
                           nn.BatchNorm1d(out),
                           nn.LeakyReLU()]
                inp = out
                out = int(out/2)
            layers += [nn.Linear(inp, n_classes),
                       nn.Dropout(0.05),
                       nn.BatchNorm1d(n_classes),
                       nn.LeakyReLU()]
              
            fc  = nn.Sequential(*layers) 
            return fc        
        
        elif fc_type == 'single':
            return nn.Linear(in_features=num_ftrs, out_features=n_classes, bias=True)    
            
        else: 
            exit(f'Error: unknown fc_type {fc_type}')


    def closest_power_of_2(self, n):
        res = 1
        while res <= n:
            prev = res
            res *= 2
        return prev
        
    '''   
    def remove_grad(self, model):
        for param in model.parameters(): 
            param.requires_grad = False
    '''
# end class Net


def run_ff(model, dataloader, device):
    # run feed-forward network
    model.eval()
    y, y_pred, outputs = None, None, None
    total_data, acc = 0, 0
    with torch.no_grad():
        for inputs, target in dataloader:
            inputs, target = inputs.to(device), target.to(device)

            output = model(inputs)
            pred = output.argmax(dim=1, keepdim=True)
            acc += pred.eq(target.view_as(pred)).sum().item()
            total_data += len(inputs)

            if y is None:
                y = target.cpu().numpy()
                y_pred = pred.cpu().numpy()
                outputs = output.cpu().numpy()
            else:
                y = np.concatenate((y, target.cpu().numpy()), axis=0)
                y_pred = np.concatenate((y_pred, pred.cpu().numpy()), axis=0)
                outputs = np.concatenate((outputs, output.cpu().numpy()), axis=0)

            # the following may help avoid CUDA memory issues:
            del inputs
            del target
            gc.collect()
            torch.cuda.empty_cache()

        accuracy = acc / total_data

    return accuracy, y, y_pred, outputs
