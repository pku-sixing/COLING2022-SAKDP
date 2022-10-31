"""
多头版的 GeneralMemoryBank
"""
import torch
from torch import nn

from finedial.modules.memory_bank.GeneralMemoryBank import GeneralMemoryBankAgent
from finedial.ops.general import model_helper

VALID_PROJECTION_FLAGS = {"key", "value", "query"}
COPY_SHARING_MODE = {
    "full": "完全共享所有的网络",
    "partial": "不共享Attention网络，但是共享投影的部分",
    "none": "完全不共享，只是使用相同的网络结构，不同的参数",
}


class HierarchicalMemoryBank(nn.Module):
    context_summary: object
    memory_bank: object
    memory_bank_length: object
    memory_bank_value: object
    max_memory_len: object
    vocab_predictor: object
    mode_selector: object

    def __init__(self, agent_name, attn_params, attn_input_param, placeholder_mode=False):
        """
            @Params:
                attn_params、attn_input_param: 分别是Context Level和Utterance的参数，
        """
        super(HierarchicalMemoryBank, self).__init__()
        self.agent_name = agent_name
        self.context_level_memory_bank = GeneralMemoryBankAgent(agent_name + '_context', attn_params['context'],
                                                                attn_input_param['context'], False)
        self.utterance_level_memory_bank = GeneralMemoryBankAgent(agent_name + '_utterance', attn_params['utterance'],
                                                                  attn_input_param['utterance'], False)

        in_dim = self.utterance_level_memory_bank.value_dim + self.context_level_memory_bank.value_dim +\
                 attn_input_param['context']['query_dim']

        self.context_utterance_selection_gate = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

        assert self.context_level_memory_bank.copy_mode == self.utterance_level_memory_bank.copy_mode
        assert self.context_level_memory_bank.has_vocab_predictor == self.utterance_level_memory_bank.has_vocab_predictor
        self.copy_mode = self.context_level_memory_bank.copy_mode
        self.has_vocab_predictor = self.context_level_memory_bank.has_vocab_predictor

    def init_memory_bank(self, context_summary, memory_bank, memory_bank_value, memory_bank_length, time_major=False):
        self.context_summary = context_summary[-1]
        self.context_level_memory_bank.init_memory_bank(memory_bank['context'], memory_bank_value['context'],
                                                        memory_bank_length['context'], time_major)
        self.utterance_level_memory_bank.init_memory_bank(memory_bank['utterance'], memory_bank_value['utterance'],
                                                          memory_bank_length['utterance'], time_major)

    def create_beams(self, beam_width):
        batch_size = self.context_summary.size()[0]
        self.context_summary = model_helper.create_beam(self.context_summary, beam_width, 0)
        self.context_level_memory_bank.create_beams(beam_width)

        # [batch, seq_len, sub_seq_len, dim]
        batch_seq_len, sub_seq_len, dim = self.utterance_level_memory_bank.memory_bank.size()
        seq_len = batch_seq_len // batch_size
        self.utterance_level_memory_bank.memory_bank = self.utterance_level_memory_bank.memory_bank.view(batch_size, seq_len, sub_seq_len, dim)
        self.utterance_level_memory_bank.memory_bank = model_helper.create_beam(self.utterance_level_memory_bank.memory_bank, beam_width, 0)
        self.utterance_level_memory_bank.memory_bank = self.utterance_level_memory_bank.memory_bank.view(batch_seq_len,
                                                                                                         sub_seq_len,
                                                                                                         dim)

        self.utterance_level_memory_bank.memory_bank_value = self.utterance_level_memory_bank.memory_bank_value.view(batch_size, seq_len, sub_seq_len, dim)
        self.utterance_level_memory_bank.memory_bank_value = model_helper.create_beam(self.utterance_level_memory_bank.memory_bank_value, beam_width, 0)
        self.utterance_level_memory_bank.memory_bank_value = self.utterance_level_memory_bank.memory_bank_value.view(batch_seq_len,
                                                                                                         sub_seq_len,
                                                                                                         dim)
        self.utterance_level_memory_bank.memory_bank_length = self.utterance_level_memory_bank.memory_bank_length.view(batch_size, seq_len)
        self.utterance_level_memory_bank.memory_bank_length = model_helper.create_beam(self.utterance_level_memory_bank.memory_bank_length, beam_width, 0)
        self.utterance_level_memory_bank.memory_bank_length = self.utterance_level_memory_bank.memory_bank_length.view(batch_seq_len)


    def access(self, query, output_distribution=False):
        assert output_distribution is False
        # Step 1, Get Context_Level Attention [batch, dim] & Distribution [batch, max_seq]
        context_readout, context_align_distribution = self.context_level_memory_bank.access(query, True)
        batch_size, max_seq_len = context_align_distribution.size()
        # [batch, dim] => [batch*max_seq, dim]
        query_for_utterance = torch.cat([query, self.context_summary], -1)
        query_for_utterance = model_helper.create_beam(query_for_utterance, max_seq_len, 0)
        # => [batch*max_seq, dim]
        utterance_readout = self.utterance_level_memory_bank.access(query_for_utterance)
        # => [batch, max_seq, dim]
        utterance_readout = utterance_readout.view(batch_size, max_seq_len, -1)
        utterance_readout = utterance_readout * context_align_distribution.unsqueeze(-1)
        utterance_readout = utterance_readout.sum(dim=1)
        concat_readout = torch.cat([utterance_readout, context_readout, query], -1)
        selection_gate = self.context_utterance_selection_gate(concat_readout)
        fused_readout = selection_gate * context_readout + (1.0 - selection_gate) * utterance_readout
        # [batch, dim]
        return fused_readout

    def generate(self, query, pad_to_max=True):
        """
            Args:
                query : [batch, query_dim]
                pad_to_max :  是否Padding到最大长度，DMF模式需要这个选项
            Returns:
                attention_vector: [batch, memory_len]
        """
        assert self.copy_mode is not False
        if pad_to_max:
            # 尚未明确如何进行分割的
            raise NotImplementedError()
        # Step 1, Get Context_Level Attention [batch, dim] & Distribution [batch, seq_len]
        context_readout, context_align_distribution = self.context_level_memory_bank.access(query, True)
        batch_size, max_seq_len = context_align_distribution.size()
        query_for_utterance = torch.cat([query, self.context_summary], -1)
        # => [batch, seq_len]
        query_for_utterance = model_helper.create_beam(query_for_utterance, max_seq_len, 0)
        # => [batch * seq_len, sub_seq_len] 这一步还不能做Pad，
        utterance_align_distribution = self.utterance_level_memory_bank.generate(query_for_utterance, False)
        #  => [batch_size, max_seq_len, sub_seq_len]
        utterance_align_distribution = utterance_align_distribution.view(batch_size, max_seq_len, -1)
        utterance_align_distribution = utterance_align_distribution * context_align_distribution.unsqueeze(-1)
        #  =>  [batch_size, max_seq_len * sub_seq_len]
        utterance_align_distribution = utterance_align_distribution.reshape(batch_size, -1)
        return utterance_align_distribution


    def get_output_dim(self):
        """
        Attention的输出维度: Context-Level的
        """
        return self.context_level_memory_bank.value_dim

    def get_gen_mode_num(self):
        """
        当前的Memory是否可以进行拷贝
        """
        if self.copy_mode or self.has_vocab_predictor:
            return 1
        else:
            return 0
