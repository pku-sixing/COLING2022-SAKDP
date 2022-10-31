"""
用语创建一些基本的RNN单元
"""
import torch
import torch.nn as nn


def create_rnn_unit(rnn_type):
    """
    得到 RNN的基本单元
    """
    if rnn_type == 'gru':
        return nn.GRU
    elif rnn_type == 'lstm':
        return nn.LSTM
    else:
        raise NotImplementedError()


def create_rnn_network_from_params(params):
    """
    创建一个RNN网络
    """
    rnn_type = params['rnn_type']
    input_size = params['input_size']
    hidden_size = params['hidden_size']
    n_layers = params['n_layers']
    dropout = params.get('dropout', 0.0)
    bidirectional = params['bidirectional']
    bias = params.get('bias', False)
    batch_first = params.get('batch_first', False)
    return create_rnn_network(rnn_type, input_size, hidden_size, n_layers, bias, batch_first, dropout, bidirectional)


def create_rnn_network(rnn_type, input_size, hidden_size, n_layers, bias, batch_first=False,
                       dropout=0.0, bidirectional=False):
    """
    创建一个RNN网络
    """
    rnn_fn = create_rnn_unit(rnn_type)
    return rnn_fn(input_size, hidden_size, n_layers, bias=bias, batch_first=batch_first,
                  dropout=dropout, bidirectional=bidirectional)
