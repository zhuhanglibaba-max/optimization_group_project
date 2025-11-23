import torch
import torch.nn as nn
from torchvision import datasets
from torch.nn import Module
from torch.optim import SGD
from model import *

def model_picker(model_name:str, num_classes:int):
    if model_name == 'ResNet18':
        model = resnet18(num_classes=num_classes)
    
    elif model_name == 'ResNet50':
        model = resnet50(num_classes=num_classes)

    elif model_name == 'ResNet101':
        model = resnet101(num_classes=num_classes)

    elif model_name == 'ResNet152':
        model = resnet152(num_classes=num_classes)
    
    elif model_name == 'VGG16':
        model = vgg16_bn(num_classes=num_classes)
 
    return model

def optimizer_picker(model_param, lr, momentum, weight_decay):
    opt = SGD(model_param, lr, momentum, weight_decay=weight_decay)
    return opt


def dataset_picker(dataset_name, root, train_transform, test_transform):
    
    if dataset_name == 'CIFAR10':
        trainset = datasets.CIFAR10(root = root, train = True, transform = train_transform, download = True)
        testset = datasets.CIFAR10(root = root, train = False, transform = test_transform, download = True)

    elif dataset_name == 'CIFAR100':
        trainset = datasets.CIFAR100(root = root, train = True, transform = train_transform, download = True)
        testset = datasets.CIFAR100(root = root, train = False, transform = test_transform, download = True)
    
    elif dataset_name == 'SVHN':
        trainset = datasets.SVHN(root = root, split = 'train', transform = train_transform, download = True)
        testset = datasets.SVHN(root = root, split = 'test', transform = test_transform, download = True)
    else:
        raise NotImplementedError('Unavaliable dataset {}!'.format(dataset_name))    
    
    return trainset, testset