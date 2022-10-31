"""
创建Projection的网络
"""
import torch
from torch import nn

from finedial.ops.network_unit import activation_helper


def create_projection_layer_from_param(param):
    in_dim = param['in_dim']
    out_dim = param['out_dim']
    bias = param.get('bias', True)
    drop_in = param.get('drop_in', 0.0)
    drop_out = param.get('drop_out', 0.0)
    activation = param['activation']
    return create_projection_layer(in_dim, out_dim, bias, drop_in, drop_out, activation)


def create_projection_layer(in_dim, out_dim, bias=True, drop_in=0.0, drop_out=0.0, activation='none'):
    network = nn.Sequential()
    if drop_in > 0:
        network.add_module('drop_in', nn.Dropout(drop_in))
    network.add_module('linear', nn.Linear(in_dim, out_dim, bias))
    if activation is not None and activation != 'none':
        network.add_module('activation_' + activation, activation_helper.create_activation_unit(activation))
    if drop_out > 0:
        network.add_module('drop_out', nn.Dropout(drop_in))
    return network


def create_vocab_prediction_layer(in_dim, vocab_size, dropout=0.0, method='default'):
    if method == 'default':
        vocab_predictor = nn.Linear(in_dim, vocab_size)
        return vocab_predictor
    elif method == 'no_bias':
        vocab_predictor = nn.Linear(in_dim, vocab_size, bias=False)
        return vocab_predictor
    elif method == 'efficient_mlp' or 'general_mlp' or 'reference_mlp' or 'reference_mlp2' or 'reference_mlp3':
        if method == 'general_mlp':
            mid_vocab_predict_size = int(in_dim)
        elif method == 'reference_mlp':
            mid_vocab_predict_size = 512
        elif method == 'reference_mlp2':
            mid_vocab_predict_size = 768
        elif method == 'reference_mlp3':
            mid_vocab_predict_size = 1280
        else:
            mid_vocab_predict_size = 1280
        vocab_predictor = nn.Sequential(
            nn.Linear(in_dim, mid_vocab_predict_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(mid_vocab_predict_size, vocab_size),
        )
        return vocab_predictor
    else:
        raise NotImplementedError()
