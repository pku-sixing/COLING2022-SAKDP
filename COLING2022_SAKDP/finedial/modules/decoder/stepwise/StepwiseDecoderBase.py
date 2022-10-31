"""
用于处理Sequential 数据的解码
Time First
"""
import torch
from torch import nn


class StepwiseDecoderBase(nn.Module):

    def __init__(self, params):
        super(StepwiseDecoderBase, self).__init__()

    def forward(self, last_state):
        pass

    def get_context(self, last_state, current_input):
        """
        获得当前时间节点下的对话上下文的查询
            Args:
                last_state: [layers, batch, dim]
        """
        # Last State 只使用最后一个layer
        # TODO 不支持LSTM
        if self.prediction_context == 'extend_full':
            n_layer, batch_size, dim = last_state.size()
            last_state = last_state.permute(1, 0, 2)
            last_state = last_state.reshape([1, batch_size, -1])
        else:
            last_state = last_state[-1:]

        contexts = []
        # Decoder的状态
        if isinstance(last_state, tuple) or isinstance(last_state, list):
            for x in last_state:
                n_layers, batch_size, dim = x.size()
                x = x.transpose(0, 1).view(batch_size, n_layers * dim)
                contexts.append(x)
        else:
            n_layers, batch_size, dim = last_state.size()
            last_state = last_state.transpose(0, 1).reshape(batch_size, n_layers * dim)
            contexts.append(last_state)
        if self.prediction_context != 'default':
            contexts.append(current_input)
        return torch.cat(contexts, -1)

    def get_memory_context(self, last_state, current_input, memory_access_readouts):
        """
        获得当前时间节点下包含所有信息的上下文
            Args:
                last_state: [layers, batch, dim]
        """
        # Last State 只使用最后一个layer
        # TODO 不支持LSTM
        if self.prediction_context == 'extend_full':
            n_layer, batch_size, dim = last_state.size()
            last_state = last_state.permute(1, 0, 2)
            last_state = last_state.reshape([1, batch_size, -1])
        else:
            last_state = last_state[-1:]

        contexts = []

        # Attention Context
        if self.prediction_context != 'default':
            contexts = contexts + memory_access_readouts

        # Decoder的状态
        if isinstance(last_state, tuple) or isinstance(last_state, list):
            for x in last_state:
                n_layers, batch_size, dim = x.size()
                x = x.transpose(0, 1).view(batch_size, n_layers * dim)
                contexts.append(x)
        else:
            n_layers, batch_size, dim = last_state.size()
            last_state = last_state.transpose(0, 1).reshape(batch_size, n_layers * dim)
            contexts.append(last_state)

        if self.prediction_context != 'default':
            contexts.append(current_input)
        return torch.cat(contexts, -1)

    def init_memory_bank(self, memory_dict):
        for agent_name in self.memory_agents.keys():
            agent = self.memory_agents[agent_name]
            memory_bank = memory_dict[agent_name]['memory_bank']
            memory_bank_value = memory_dict[agent_name]['memory_bank_value']
            memory_bank_length = memory_dict[agent_name]['memory_bank_length']
            time_major = memory_dict[agent_name]['time_major']
            agent.init_memory_bank(memory_bank, memory_bank_value, memory_bank_length, time_major=time_major)

    def _update_decoder_state(self, last_state, current_input, agent_pooling='concat'):

        # 获取当前的Access 状态
        memory_access_readouts = []
        if len(self.memory_agents) > 0:
            memory_access_query = self.get_context(last_state, current_input)
            for agent_name in self.memory_agents.keys():
                agent = self.memory_agents[agent_name]
                readout = agent.access(memory_access_query)
                memory_access_readouts.append(readout)

        if agent_pooling == 'mean':
            tmp = memory_access_readouts[0]
            for i in range(1, len(memory_access_readouts)):
                tmp = tmp + memory_access_readouts[i]
            tmp = tmp / len(memory_access_readouts)
            memory_access_readouts = [tmp]

        # 更新当前的Decoder State
        decoder_input_state = [current_input] + memory_access_readouts
        decoder_input_state = torch.cat(decoder_input_state, -1)
        decoder_input_state = decoder_input_state.unsqueeze(0)
        if last_state is not None:
            last_state = last_state.contiguous()
        decoder_output, last_state = self.rnn(decoder_input_state, last_state)

        # 计算当前的概率
        current_dialogue_context = self.get_memory_context(last_state, current_input, memory_access_readouts)

        return decoder_output, last_state, current_dialogue_context

    def _compute_vocab_prob_dist(self, current_dialogue_context):
        gen_vocab_logits = self.vocab_generation_mode_predictor(current_dialogue_context)
        gen_vocab_prob_dist = torch.softmax(gen_vocab_logits, -1)
        return gen_vocab_prob_dist
