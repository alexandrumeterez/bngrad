import torchvision.transforms as transforms
from modules.models import SinAct
from torch import nn
import numpy as np

# Constants
DS_INPUT_SIZES = {
    'CIFAR10': 3 * 32 * 32,
    'CIFAR100': 3 * 32 * 32,
    'MNIST': 28 * 28,
    'FashionMNIST': 28 * 28,
}
DS_NUM_CLASSES = {
    'CIFAR10': 10,
    'CIFAR100': 100,
    'MNIST': 10,
    'FashionMNIST': 10,
}

DS_TRANSFORMS = {
    'CIFAR10': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]),
    'CIFAR100': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]),
    'MNIST': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]),
    'FashionMNIST': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860), (0.3530))]),
}

ACTIVATIONS = {
    'identity': nn.Identity,
    'sin': SinAct,
    'tanh': nn.Tanh,
    'selu': nn.SELU,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU
}

GAINS = {
    'identity': 1,
    'sin': 1,
    'tanh': 5/3,
    'selu': 3/4,
    'relu': np.sqrt(2),
    'leaky_relu': nn.init.calculate_gain('leaky_relu')
}