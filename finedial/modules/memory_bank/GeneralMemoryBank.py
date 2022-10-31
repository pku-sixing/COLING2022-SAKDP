from finedial.ops.attention.general.GeneralAttention import GeneralAttention
import torch
from torch import nn

from finedial.ops.general import model_helper
from finedial.ops.network_unit import projection_helper

VALID_PROJECTION_FLAGS = {"key", "value", "query"}
COPY_SHARING_MODE = {
    "full": "完全共享所有的网络",
    "partial": "不共享Attention网络，但是共享投影的部分",
    "none": "完全不共享，只是使用相同的网络结构，不同的参数",
}


class GeneralMemoryBankAgent(nn.Module):
    """
        管理常见的Attention操作。
    """
    memory_bank: object
    memory_bank_length: object
    memory_bank_value: object
    max_memory_len: object
    vocab_predictor: object
    mode_selector: object

    def __init__(self, agent_name, attn_params, attn_input_param, placeholder_mode=False):
        super(GeneralMemoryBankAgent, self).__init__()
        self.agent_name = agent_name
        self.placeholder_mode = placeholder_mode
        self.attention_configs = attn_params
        self.max_copy_candidate_nums = attn_input_param.max_copy_candidate_nums
        attention_type = attn_params.attention_type
        query_dim = attn_input_param.query_dim
        key_dim = attn_input_param.key_dim
        value_dim = attn_input_param.value_dim
        hidden_dim = attn_params.hidden_size
        copy_mode = attn_params.copy_mode

        self.copy_mode = copy_mode
        self.has_vocab_predictor = False

        # 允许进行特定的Projection
        self.query_projection = None
        self.key_projection = None
        self.value_projection = None

        if attn_params.get('projection_flags', None) is None:
            projection_flags = set()
        else:
            projection_flags = set(attn_params.get('projection_flags').keys())
        assert len(projection_flags - VALID_PROJECTION_FLAGS) == 0, \
            'Invalid projection_flags: %s' % attn_params.projection_flags

        if 'query' in projection_flags:
            item = 'query'
            attn_params.projection_flags[item]['in_dim'] = query_dim
            self.query_projection = projection_helper.create_projection_layer_from_param(
                attn_params.projection_flags[item])
            query_dim = attn_params.projection_flags[item]['out_dim']

        if 'key' in projection_flags:
            item = 'key'
            attn_params.projection_flags[item]['in_dim'] = key_dim
            self.key_projection = projection_helper.create_projection_layer_from_param(
                attn_params.projection_flags[item])
            key_dim = attn_params.projection_flags[item]['out_dim']

        if 'value' in projection_flags:
            item = 'value'
            attn_params.projection_flags[item]['in_dim'] = value_dim
            self.value_projection = projection_helper.create_projection_layer_from_param(
                attn_params.projection_flags[item])
            value_dim = attn_params.projection_flags[item]['out_dim']

        self.value_dim = value_dim
        self.attention_network = GeneralAttention(attention_type, query_dim, key_dim, value_dim, hidden_dim)

        if self.copy_mode:
            share_mode = self.copy_mode.share_network_with_attention
            assert share_mode in COPY_SHARING_MODE
            if share_mode == 'full':
                self.copy_network = self.attention_network
            elif share_mode == 'partial':
                self.copy_network = GeneralAttention(attention_type, query_dim, key_dim, value_dim, hidden_dim)
            else:
                raise NotImplementedError()
        else:
            self.copy_network = None

    def create_vocab_predictor(self, input_size, hidden_size, fixed_vocab_size, dropout, vocab_predictor_mode="efficient_mlp"):
        self.has_vocab_predictor = True
        self.vocab_predictor = projection_helper.create_vocab_prediction_layer(
            input_size, fixed_vocab_size, dropout=dropout,
            method=vocab_predictor_mode)
        if self.copy_mode:
            self.mode_selector = nn.Linear(input_size, 2, bias=True)
        else:
            self.mode_selector = None

    def generate_vocab(self, query, state, dynamic_projections=None, pad_to_max=True):
        """
            Args:
                query : [batch, last_word + state_dim]
                pad_to_max :  是否Padding到最大长度，DMF模式需要这个选项
            Returns:
                attention_vector: [batch, memory_len]
        """
        assert not self.placeholder_mode

        gen_vocab_logits = self.vocab_predictor(torch.cat([query, state], -1))
        gen_vocab_probs = torch.softmax(gen_vocab_logits, -1)

        if self.copy_mode:
            if self.query_projection is not None:
                copy_query = self.query_projection(query)
            else:
                copy_query = query
            attention_vector, align_distributions = self.copy_network(copy_query, self.memory_bank, self.memory_bank_value,
                                                                      self.memory_bank_length, valid_bank_start_index=1)

            if pad_to_max:
                batch_size, src_len = align_distributions.size()
                if src_len < self.max_copy_candidate_nums:
                    padding = torch.zeros([batch_size, self.max_copy_candidate_nums - src_len],
                                          device=align_distributions.device, dtype=align_distributions.dtype)
                    align_distributions = torch.cat([align_distributions, padding], dim=-1)
                elif src_len == self.max_copy_candidate_nums:
                    pass
                else:
                    raise IndexError()

            if dynamic_projections is not None:
                copy_projection_matrix = dynamic_projections[self.agent_name]
                tmp_copy = align_distributions.unsqueeze(1)
                copy_prob = torch.bmm(tmp_copy, copy_projection_matrix).squeeze(1)
                modes = torch.softmax(self.mode_selector(torch.cat([query, state], -1)), -1)
                padding_size = copy_prob.size()[1] - gen_vocab_probs.size()[1]
                batch_size = copy_prob.size()[0]
                if padding_size > 0:
                    gen_vocab_probs = torch.cat([gen_vocab_probs, torch.zeros([batch_size, padding_size],
                                                                              device=gen_vocab_probs.device)], dim=-1)
                res = modes[:, 0:1] * gen_vocab_probs + modes[:, 1:2] * copy_prob
                return [res]
            else:
                copy_prob = align_distributions
                raise NotImplementedError()


        else:
            return [gen_vocab_probs]


    # 独立封装成一个Memory Bank
    def init_memory_bank(self, memory_bank, memory_bank_value, memory_bank_length, time_major=False):
        self.memory_bank = memory_bank
        self.memory_bank_length = memory_bank_length
        self.max_memory_len = memory_bank.size()[0]
        if self.placeholder_mode:
            return
        if time_major:
            memory_bank = memory_bank.transpose(1, 0)
            memory_bank_value = memory_bank_value.transpose(1, 0)
        else:
            memory_bank = memory_bank

        if self.key_projection is not None:
            self.memory_bank = self.key_projection(memory_bank)
        else:
            self.memory_bank = memory_bank

        if self.value_projection is not None:
            self.memory_bank_value = self.value_projection(memory_bank_value)
        else:
            self.memory_bank_value = memory_bank_value

    def create_beams(self, beam_width):
        if self.placeholder_mode:
            return
        self.memory_bank = model_helper.create_beam(self.memory_bank, beam_width, 0)
        self.memory_bank_value = model_helper.create_beam(self.memory_bank_value, beam_width, 0)
        self.memory_bank_length = model_helper.create_beam(self.memory_bank_length, beam_width, 0)

    def get_output_dim(self):
        """
        Attention的输出维度
        """
        return self.value_dim

    def get_gen_mode_name(self):
        if self.copy_mode:
            return [self.agent_name]
        else:
            return []

    def get_gen_mode_num(self):
        """
        当前的Memory是否可以进行拷贝
        """
        if self.copy_mode or self.has_vocab_predictor:
            return 1
        else:
            return 0

    def access(self, query, output_distribution=False):
        """
            Args:
                query : [batch, query_dim]
                output_distribution :
            Returns:
                attention_vector: [batch, out_dim]
        """
        if not self.placeholder_mode:
            if self.query_projection is not None:
                query = self.query_projection(query)
            attention_vector, align_distributions = self.attention_network(query, self.memory_bank,
                                                                           self.memory_bank_value,
                                                                           self.memory_bank_length)
            if output_distribution:
                return attention_vector, align_distributions
            else:
                return attention_vector
        else:
            assert output_distribution is False
            batch_size = query.size()[0]
            out_dim = self.get_output_dim()
            return torch.zeros([batch_size, out_dim], device=query.device, requires_grad=False)

    def generate(self, query, pad_to_max=True):
        """
            Args:
                query : [batch, query_dim]
                pad_to_max :  是否Padding到最大长度，DMF模式需要这个选项
            Returns:
                attention_vector: [batch, memory_len]
        """
        assert self.copy_mode is not False
        if not self.placeholder_mode:
            if self.query_projection is not None:
                query = self.query_projection(query)
            attention_vector, align_distributions = self.copy_network(query, self.memory_bank, self.memory_bank_value,
                                                                      self.memory_bank_length, valid_bank_start_index=1)
        else:
            batch_size = query.size()[0]
            memory_len = self.max_memory_len
            align_distributions = torch.zeros([batch_size, memory_len], device=query.device, requires_grad=False)

        if pad_to_max:
            batch_size, src_len = align_distributions.size()
            if src_len < self.max_copy_candidate_nums:
                padding = torch.zeros([batch_size, self.max_copy_candidate_nums - src_len],
                                      device=align_distributions.device, dtype=align_distributions.dtype)
                align_distributions = torch.cat([align_distributions, padding], dim=-1)
            elif src_len == self.max_copy_candidate_nums:
                pass
            else:
                raise IndexError()
        return align_distributions
