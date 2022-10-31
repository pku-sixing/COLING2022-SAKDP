"""
用于处理Sequential 数据的解码
Time First
"""
import torch
from torch import nn

from finedial.modules.decoder.stepwise.StepwiseDecoderBase import StepwiseDecoderBase
from finedial.modules.decoder.stepwise.StepwiseDecoderState import StepwiseDecoderState
from finedial.modules.memory_bank.MultiHeadGeneralMemoryBank import MultiHeadGeneralMemoryBankAgent
from finedial.ops.network_unit import rnn_helper, projection_helper


class StepwiseRNNDecoder(StepwiseDecoderBase):

    def __init__(self, params, memory_agents={}):
        super(StepwiseRNNDecoder, self).__init__(params)
        self.decoder_type = 'rnn'
        self.hidden_size = params.hidden_size
        self.input_size = params.input_size
        self.n_layers = params.n_layers
        self.rnn_type = params.rnn_type
        self.fixed_vocab_size = params.fixed_vocab_size
        self.dropout = params.dropout
        self.vocab_predictor_mode = params.vocab_predictor_mode
        self.prediction_context = params.get('prediction_context', 'extend')
        self.generation_mode_fusion_mode = params.generation_mode_fusion_mode
        self.generation_mode_gate_mode = params.generation_mode_gate_mode
        self.extend_vocab_size = params.extend_vocab_size

        # 配置Memory Agent
        memory_generation_mode_num = 0
        self.memory_readout_dim = 0
        self.memory_agents = memory_agents
        self.memory_agent_order = [None]
        for agent_name, agent in memory_agents.items():
            self.memory_readout_dim += agent.get_output_dim()
            memory_generation_mode_num += agent.get_gen_mode_num()
            if isinstance(agent, MultiHeadGeneralMemoryBankAgent):
                for key in agent.heads.keys():
                    self.memory_agent_order.append(agent_name + '_' + key)
            else:
                self.memory_agent_order.append(agent_name)

        # 开始计算Decoder输入维度: [LastToken, Attention]
        self.decoder_input_dim = self.input_size + self.memory_readout_dim

        # Knowledge Attention Memory的 [Last Token, Decoder_State]
        self.memory_input_dim = self.input_size + self.hidden_size
        if params.rnn_type == 'lstm':
            self.memory_input_dim += self.hidden_size

        # Decoder更新的维度大小，不包含上一个的状态，Last Input + MemoryReadouts
        self.state_input_dim = self.input_size + self.memory_readout_dim

        # Decoder 输出的维度，用于预测的维度， Last Input + MemoryReadouts + Last Decoder State
        if self.prediction_context == 'extend_full':
            self.vocab_prediction_in_dim = self.input_size + self.memory_readout_dim + self.hidden_size * self.n_layers
        elif self.prediction_context == 'extend':
            self.vocab_prediction_in_dim = self.input_size + self.memory_readout_dim + self.hidden_size
        elif self.prediction_context == 'default':
            self.vocab_prediction_in_dim = self.hidden_size

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
            self.vocab_prediction_in_dim, self.fixed_vocab_size, dropout=self.dropout, method=self.vocab_predictor_mode
        )

        # 开始构造词表的输出预测
        if memory_generation_mode_num > 0:
            self.generation_mode_num = memory_generation_mode_num + 1
            if self.generation_mode_gate_mode == 'mlp':
                self.generation_mode_selector = nn.Sequential(
                    nn.Linear(self.vocab_prediction_in_dim, self.hidden_size, bias=True),
                    torch.nn.ELU(),
                    nn.Linear(self.hidden_size, self.generation_mode_num, bias=True),
                )
            elif self.generation_mode_gate_mode == 'std':
                self.generation_mode_selector = nn.Linear(self.vocab_prediction_in_dim, self.generation_mode_num, bias=True)
            else:
                raise NotImplementedError()
        else:
            self.generation_mode_num = 1
            assert self.generation_mode_gate_mode == 'none'

    def forward(self, decoder_state):
        """
            Inputs:
                last_state:
                    state: 当前的decoder状态信息
                    current_input: [batch_size, input_dim]当前的decoder输入
                    dynamic_vocab_projections: 投影到动态词表上
        """

        assert isinstance(decoder_state, StepwiseDecoderState)
        last_state = decoder_state.state
        current_input = decoder_state.input

        decoder_output, last_state, current_dialogue_context = self._update_decoder_state(last_state, current_input)
        gen_vocab_prob_dist = self._compute_vocab_prob_dist(current_dialogue_context)

        # 拷贝的概率
        pad2max = self.generation_mode_fusion_mode == 'dynamic_mode_fusion'
        gen_mode_names = ['vocab']

        # 优化并确认拷贝的概率模式
        if self.generation_mode_num > 1:
            local_copy_prob_dists = []
            memory_generate_query = self.get_context(last_state, current_input)
            for agent_name in self.memory_agents.keys():
                agent = self.memory_agents[agent_name]
                if agent.copy_mode is False:
                    continue
                local_copy_prob = agent.generate(memory_generate_query, pad_to_max=pad2max)
                if not isinstance(local_copy_prob, list):
                    local_copy_prob = [local_copy_prob]
                agent_names = agent.get_gen_mode_name()
                if isinstance(agent_names, str):
                    agent_names = [agent_names]
                gen_mode_names += agent_names
                local_copy_prob_dists += local_copy_prob

            # 合并
            generation_mode_gate_logits = self.generation_mode_selector(current_dialogue_context)
            generation_mode_gate = torch.softmax(generation_mode_gate_logits, -1)
            local_prob_dist = [gen_vocab_prob_dist] + local_copy_prob_dists
            local_prob_dist = [local_prob_dist[idx] * generation_mode_gate[:, idx:idx + 1]
                               for idx in range(self.generation_mode_num)]

        else:
            local_prob_dist = [gen_vocab_prob_dist]

        decoder_state.token_prob_dist = [x.unsqueeze(0) for x in local_prob_dist]
        decoder_state.token_mode_order = gen_mode_names
        decoder_state.state = last_state
        return decoder_state
