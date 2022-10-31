"""
用于处理Sequential 数据的解码
Time First
"""
import torch
from torch import nn

from finedial.modules.decoder.stepwise.StepwiseDecoderBase import StepwiseDecoderBase
from finedial.modules.memory_bank.MultiHeadGeneralMemoryBank import MultiHeadGeneralMemoryBankAgent
from finedial.ops.network_unit import rnn_helper, projection_helper


class StepwiseRNNDecoderSingleChannel(StepwiseDecoderBase):

    def __init__(self, params, memory_agents={}):
        super(StepwiseRNNDecoderSingleChannel, self).__init__(params)
        self.decoder_type = 'rnn'
        self.hidden_size = params.hidden_size
        self.input_size = params.input_size
        self.n_layers = params.n_layers
        self.rnn_type = params.rnn_type
        self.fixed_vocab_size = params.fixed_vocab_size
        self.dropout = params.dropout
        self.vocab_predictor_mode = params.vocab_predictor_mode
        self.generation_mode_fusion_mode = params.generation_mode_fusion_mode
        self.generation_mode_gate_mode = params.generation_mode_gate_mode
        self.extend_vocab_size = params.extend_vocab_size
        assert self.predict_with_full_states is False, 'do not implement this now'

        # 配置Memory Agent
        self.memory_readout_dim = 0
        memory_generation_mode_num = 0
        self.memory_agents = memory_agents
        self.memory_agent_order = [None]
        for agent_name in memory_agents.keys():
            agent = memory_agents[agent_name]
            if self.memory_readout_dim == 0:
                self.memory_readout_dim += agent.get_output_dim()
            else:
                assert self.memory_readout_dim == agent.get_output_dim()
            memory_generation_mode_num += agent.get_gen_mode_num()
            if isinstance(agent, MultiHeadGeneralMemoryBankAgent):
                for key in agent.heads.keys():
                    self.memory_agent_order.append(agent_name+'_'+key)
            else:
                self.memory_agent_order.append(agent_name)

        # 开始计算各种输入维度
        self.decoder_input_dim = self.input_size + self.memory_readout_dim
        # Knowledge Attention Memory的
        self.memory_input_dim = self.input_size + self.hidden_size
        if params.rnn_type == 'lstm':
            self.memory_input_dim += self.hidden_size
        # Decoder更新的维度大小，不包含上一个的状态，Last Input + MemoryReadouts
        self.state_input_dim = self.input_size + self.memory_readout_dim
        # Decoder 输出的维度，用于预测的维度， Last Input + MemoryReadouts + Last Decoder State
        self.state_output_dim = self.input_size + self.memory_readout_dim + self.hidden_size

        # 配置Decoder的GRU
        rnn_type = params.rnn_type
        input_size = self.decoder_input_dim
        hidden_size = params.hidden_size
        n_layers = params.n_layers
        dropout = params.dropout
        self.rnn = rnn_helper.create_rnn_network(rnn_type, input_size, hidden_size, n_layers, bias=True,
                                                 dropout=dropout)

        # 开始构建词表预测输出
        self.vocab_generation_mode_predictor = projection_helper.create_vocab_prediction_layer(
            self.state_output_dim, self.fixed_vocab_size, dropout=self.dropout, method=self.vocab_predictor_mode
        )
        # 开始构造词表的输出预测
        if memory_generation_mode_num > 0:
            self.generation_mode_num = memory_generation_mode_num + 1
            if self.generation_mode_gate_mode == 'mlp':
                self.generation_mode_selector = nn.Sequential(
                    nn.Linear(self.state_output_dim, self.hidden_size, bias=True),
                    torch.nn.ELU(),
                    nn.Linear(self.hidden_size, self.generation_mode_num, bias=False),
                )
            elif self.generation_mode_gate_mode == 'std':
                self.generation_mode_selector = nn.Linear(self.state_output_dim, self.generation_mode_num, bias=False)
            else:
                raise NotImplementedError()
        else:
            self.generation_mode_num = 1
            assert self.generation_mode_gate_mode == 'none'


    def forward(self, last_state, current_input, memory_dict, dynamic_projections):
        """
            Inputs:
                last_state: 当前的decoder状态信息
                current_input: [batch_size, input_dim]当前的decoder输入
                memory_dict: 所有固定的memory_bank
                    attentive_memory_banks: 所有通过标准Attention方式访问的 Attentions
                dynamic_projections: 投影到动态词表上
        """
        batch_size, input_dim = current_input.size()

        decoder_output, last_state, current_dialogue_context = self._update_decoder_state(last_state, current_input,
                                                                                          agent_pooling='mean')
        gen_vocab_probs = self._compute_vocab_prob_dist(current_dialogue_context)
        
        # 拷贝的概率
        if self.generation_mode_num > 1:
            local_copy_probs = []
            memory_generate_query = self.get_context(last_state, current_input, with_memory=False)
            for agent_name in self.memory_agents.keys():
                agent = self.memory_agents[agent_name]
                if agent.copy_mode is False:
                    continue
                local_copy_prob = agent.generate(memory_generate_query,
                                                 pad_to_max=self.generation_mode_fusion_mode == 'dynamic_mode_fusion')
                if not isinstance(local_copy_prob, list):
                    local_copy_prob = [local_copy_prob]
                local_copy_probs += local_copy_prob

            # 合并
            generation_mode_gate_logits = self.generation_mode_selector(current_dialogue_context)
            generation_mode_gate = torch.softmax(generation_mode_gate_logits, -1)
            local_probs = [gen_vocab_probs] + local_copy_probs
            local_probs = [local_probs[idx] * generation_mode_gate[:, idx:idx + 1]
                           for idx in range(self.generation_mode_num)]
            if self.generation_mode_fusion_mode == 'dynamic_mode_fusion':
                token_probs = torch.cat(local_probs, -1)
            else:
                padding_size = self.extend_vocab_size
                if padding_size > 0:
                    token_probs = torch.cat([local_probs[0], torch.zeros([batch_size, padding_size],
                                                                         device=local_probs[0].device)], dim=-1)
                else:
                    token_probs = local_probs[0]
                for local_index in range(1, len(local_probs)):
                    copy_projection_matrix = dynamic_projections[self.memory_agent_order[local_index]]
                    tmp_copy = local_probs[local_index].unsqueeze(1)
                    tmp_prob = torch.bmm(tmp_copy, copy_projection_matrix).squeeze(1)
                    token_probs += tmp_prob
        else:
            token_probs = gen_vocab_probs

        return torch.log(token_probs + 1e-20), last_state
