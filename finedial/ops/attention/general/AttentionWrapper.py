"""
用于包裹Attention的基本操作
"""
import torch
from torch import nn


class AttentionWrapper(nn.Module):

    def __init__(self, params, ):
        super(AttentionWrapper, self).__init__()
