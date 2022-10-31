"""
多头版的 GeneralMemoryBank
"""
import torch
from torch import nn

from finedial.modules.memory_bank.GeneralMemoryBank import GeneralMemoryBankAgent


class MultiHeadGeneralMemoryBankAgent(nn.Module):
    memory_bank: object
    memory_bank_length: object
    memory_bank_value: object

    def __init__(self, agent_name, attn_params, attn_input_param, head_names, placeholder_mode):
        num_heads = len(head_names)
        assert num_heads > 1, 'head number must larger than 1'
        super(MultiHeadGeneralMemoryBankAgent, self).__init__()
        heads = [GeneralMemoryBankAgent(agent_name+'_'+head_name, attn_params, attn_input_param, placeholder_mode)
                 for head_name in head_names]
        self.heads = nn.ModuleDict()
        self.copy_mode = False
        self.placeholder_mode = False
        for head_name, head in zip(head_names, heads):
            self.heads[head_name] = head
            self.copy_mode = head.copy_mode
            self.placeholder_mode = head.placeholder_mode

    def init_memory_bank(self, memory_bank, memory_bank_value, memory_bank_length, time_major=False):
        for head_name in self.heads:
            head = self.heads[head_name]
            head.init_memory_bank(memory_bank, memory_bank_value, memory_bank_length, time_major)

    def create_beams(self, beam_width):
        for head_name in self.heads:
            head = self.heads[head_name]
            head.create_beams(beam_width)

    def get_output_dim(self):
        """
        Attention的输出维度
        """
        res = 0
        for head_name in self.heads:
            head = self.heads[head_name]
            res += head.get_output_dim()
        return res


    def get_gen_mode_name(self):
        """
        当前的Memory是否可以进行拷贝
        """
        res = []
        for head_name in self.heads:
            head = self.heads[head_name]
            res += head.get_gen_mode_name()
        return res

    def get_gen_mode_num(self):
        """
        当前的Memory是否可以进行拷贝
        """
        res = 0
        for head_name in self.heads:
            head = self.heads[head_name]
            res += head.get_gen_mode_num()
        return res

    def access(self, query, output_distribution=False):
        """
            Args:
                query : [batch, query_dim]
                output_distribution :
            Returns:
                attention_vector:
        """
        res = []
        distributions = []
        for head_name in self.heads:
            head = self.heads[head_name]
            if output_distribution:
                tmp = head.access(query, output_distribution)
                res.append(tmp[0])
                distributions.append(tmp[1])
            else:
                res.append(head.access(query))

        if output_distribution:
            return torch.cat(res, -1), distributions
        else:
            return torch.cat(res, -1)

    def generate(self, query, pad_to_max=True):
        res = []
        for head_name in self.heads:
            head = self.heads[head_name]
            res.append(head.generate(query, pad_to_max))
        return res

    def create_vocab_predictor(self, input_size, hidden_size, fixed_vocab_size, dropout, vocab_predictor_mode):
        for head in self.heads.values():
            head.create_vocab_predictor(input_size, hidden_size, fixed_vocab_size, dropout, vocab_predictor_mode)

    def generate_vocab(self, query, state, dynamic_projections=None, pad_to_max=True):
        res = []
        for head in self.heads.values():
            tmp = head.generate_vocab(query, state, dynamic_projections, pad_to_max)
            res += tmp
        return res