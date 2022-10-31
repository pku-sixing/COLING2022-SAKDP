import math
import time
from collections import defaultdict

import torch
from torch import nn

from finedial.utils.logging.logger_helper import logger


class BaseWrapper:
    """
       用来执行和控制一个Seq2Seq模型的

       如果模型彼此直接没有的参数，那么建议一个模型一个FineTrainer

    """
    def __init__(self, model_dict, optimizer_dict, params_dict):
        """
            model_dict: 所有装载的模型
        """
        if isinstance(model_dict, nn.Module):
            model_dict = {"major_model": model_dict}
            optimizer_dict = {"major_model": optimizer_dict}
            params_dict = {"major_model": params_dict}

        assert model_dict.keys() == optimizer_dict.keys()
        assert model_dict.keys() == params_dict.keys()

        self.model_dict = model_dict
        self.optimizer_dict = optimizer_dict
        self.params_dict = params_dict

        self.model_names = list(model_dict.keys())
        self.models = list(model_dict.values())
        self.model_items = list(model_dict.items())
        # for model_name, model in self.model_dict.items():
        #     assert isinstance(model, Seq2Seq) or isinstance(model, Ref2Seq)

    def eval(self):
        for model_name, model in self.model_items:
            model.eval()
            logger.info("[Model] Set %s to EVAL mode" % model_name)

    def train(self):
        for model_name, model in self.model_items:
            model.train()
            logger.info("[Model] Set %s to TRAIN mode" % model_name)
