import torch.nn as nn


def build_embedding(token_num, embed_size):
    return nn.Embedding(token_num, embed_size)


def build_embedding_from_param(param):
    return build_embedding(param['token_vocab_size'], param['embed_size'])

