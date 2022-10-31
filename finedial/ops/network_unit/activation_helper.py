"""
创造激活层
"""
import torch

def create_activation_func(method):
    if method == 'tanh':
        return torch.nn.Tanh
    elif method == 'sigmoid':
        return torch.nn.Sigmoid
    elif method == 'relu':
        return torch.nn.ReLU
    else:
        raise NotImplementedError()


def create_activation_unit(method):
    func = create_activation_func(method)
    return func()
