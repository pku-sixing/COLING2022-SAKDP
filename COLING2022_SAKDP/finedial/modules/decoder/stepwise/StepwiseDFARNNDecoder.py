"""
用于处理Sequential 数据的解码
Time First
"""
import torch
from torch import nn

from finedial.modules.decoder.stepwise.StepwiseDecoderBase import StepwiseDecoderBase
from finedial.ops.network_unit import rnn_helper, projection_helper


class StepwiseDFARNNDecoder(StepwiseDecoderBase):
    """
        Diffuse-Aggregate 模式的
    """

    def __init__(self, params, memory_agents={}):
        super(StepwiseDFARNNDecoder, self).__init__(params)
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
        self.stop_agent_flags = set(params.get("stop_agent_flags", []))

        self.sequential_input_size = self.input_size + self.hidden_size
        self.sequential_hidden_size = self.hidden_size
        self.sequential_state_size = self.hidden_size
        self.sequential_topics = []

        self.use_vocab_agent = params.get('use_vocab_agent', True)
        if self.use_vocab_agent is False:
            self.sequential_input_size -= self.hidden_size

        for agent_name, agent in memory_agents.items():
            assert agent.placeholder_mode is False
            if agent_name in self.stop_agent_flags:
                continue
            self.sequential_input_size += self.hidden_size
            if agent_name != 'query' and params.get("extended_state_manager", False):
                self.sequential_hidden_size += self.hidden_size // 2

            if 'topic_input_flags' in params and agent_name in params.topic_input_flags:
                self.sequential_input_size += agent.get_output_dim()
                self.sequential_topics.append(agent_name)

        # 配置Decoder的GRU
        rnn_type = params.rnn_type
        hidden_size = params.hidden_size
        n_layers = params.n_layers
        dropout = params.dropout

        # 配置Sequential Manager
        self.sequential_state_manager = rnn_helper.create_rnn_network('gru', self.sequential_input_size,
                                                                      self.sequential_hidden_size, n_layers,
                                                                      bias=True, dropout=self.dropout)

        if self.sequential_hidden_size != self.sequential_state_size:
            self.sequential_state_projection = projection_helper.create_projection_layer(self.sequential_state_size,
                                                                                         self.sequential_hidden_size,
                                                                                         activation='tanh')
            self.sequential_state_down_scale = projection_helper.create_projection_layer(self.sequential_hidden_size,
                                                                                         self.sequential_state_size,
                                                                                         activation='tanh')
        else:
            self.sequential_state_projection = lambda x: x
            self.sequential_state_down_scale = lambda x: x

        # Decoders
        self.decoder_input_dim = dict()
        self.decoders = nn.ModuleDict()

        # 仅读取input_size
        if self.use_vocab_agent:
            self.decoder_input_dim['vocab'] = self.input_size + self.sequential_state_size
            self.decoders['vocab'] = rnn_helper.create_rnn_network(rnn_type, self.decoder_input_dim['vocab'],
                                                                   hidden_size, n_layers, bias=True, dropout=dropout)

        # MemoryAgents Decoder
        for agent_name in memory_agents.keys():
            if agent_name in self.stop_agent_flags:
                continue
            agent = memory_agents[agent_name]
            self.decoder_input_dim[agent_name] = self.input_size + agent.get_output_dim() + self.sequential_state_size
            self.decoders[agent_name] = rnn_helper.create_rnn_network(rnn_type, self.decoder_input_dim[agent_name],
                                                                      hidden_size, n_layers, bias=True, dropout=dropout)
            # Create Vocabs Predictor:
            # sequential_state + last_token
            agent.create_vocab_predictor(self.sequential_state_size + self.input_size + self.hidden_size,
                                         self.hidden_size,
                                         self.fixed_vocab_size, self.dropout, self.vocab_predictor_mode)
        # # 配置Memory Agent
        self.memory_agents = memory_agents

        # 配置Decoder的GRU
        rnn_type = params.rnn_type
        hidden_size = params.hidden_size
        n_layers = params.n_layers
        dropout = params.dropout
        self.rnn = nn.ModuleDict()
        if self.use_vocab_agent:
            self.rnn['vocab'] = rnn_helper.create_rnn_network(rnn_type, self.input_size + self.sequential_state_size,
                                                              hidden_size, n_layers, bias=True,
                                                              dropout=dropout)
        for agent_name, input_dim in self.decoder_input_dim.items():
            if agent_name in self.stop_agent_flags:
                continue
            self.rnn[agent_name] = rnn_helper.create_rnn_network(rnn_type, input_dim, hidden_size, n_layers, bias=True,
                                                                 dropout=dropout)

        # Last Word
        if self.use_vocab_agent:
            self.vocab_predictor = projection_helper.create_vocab_prediction_layer(
                self.input_size + self.sequential_state_size, self.fixed_vocab_size, dropout=self.dropout,
                method="efficient_mlp")

        # Global Mode Selector
        generation_mode_num = 1 if self.use_vocab_agent else 0
        for agent_name, x in memory_agents.items():
            if agent_name in self.stop_agent_flags:
                continue
            generation_mode_num += x.get_gen_mode_num()

        if self.generation_mode_gate_mode == 'mlp':
            self.global_mode_selector = nn.Sequential(
                nn.Linear(self.input_size + self.sequential_state_size, self.hidden_size, bias=True),
                torch.nn.ELU(),
                nn.Linear(self.hidden_size, generation_mode_num, bias=False),
            )
        elif self.generation_mode_gate_mode == 'std':
            self.global_mode_selector = nn.Linear(self.input_size + self.sequential_state_size, generation_mode_num,
                                                  bias=False)

    def _get_aggregated_last_state(self, last_state):
        aggregated_decoder_output = last_state['vocab'][-1:]
        for agent_name in self.memory_agents.keys():
            aggregated_decoder_output += last_state[agent_name][-1:]
        return aggregated_decoder_output

    def _update_decoder_state(self, last_state, last_global_state, current_input, memory_dict):
        batch_size, input_dim = current_input.size()
        if torch.is_tensor(last_state):
            # First state
            new_last_state = dict()
            if self.use_vocab_agent:
                new_last_state['vocab'] = last_state
            for agent_name in self.memory_agents.keys():
                new_last_state[agent_name] = last_state + 0.0
            last_state = new_last_state
            last_global_state = self.sequential_state_projection(last_global_state)

        # 获取当前的Access 状态
        decoder_output = dict()
        memory_access_query = torch.cat([self.sequential_state_down_scale(last_global_state[-1]), current_input], -1)
        query = memory_access_query.unsqueeze(0)

        decoder_outputs = []
        last_states = []
        if self.use_vocab_agent:
            vocab_decoder_output, vocab_last_state = self.rnn['vocab'](query,
                                                                       last_state['vocab'].contiguous())
            last_state['vocab'] = vocab_last_state
            decoder_output['vocab'] = vocab_decoder_output
            sequential_inputs = [current_input.unsqueeze(0), vocab_decoder_output]

            decoder_outputs.append(vocab_decoder_output)
            last_states.append(vocab_last_state)
        else:
            sequential_inputs = [current_input.unsqueeze(0)]
        if len(self.memory_agents) > 0:
            for agent_name in self.memory_agents.keys():
                if agent_name in self.stop_agent_flags:
                    continue
                agent = self.memory_agents[agent_name]
                readout = agent.access(memory_access_query)

                # 更新当前的Decoder State
                decoder_input_state = [current_input, readout, self.sequential_state_down_scale(last_global_state[-1])]
                decoder_input_state = torch.cat(decoder_input_state, -1)
                decoder_input_state = decoder_input_state.unsqueeze(0)
                mem_decoder_output, mem_last_state = self.rnn[agent_name](decoder_input_state,
                                                                          last_state[agent_name].contiguous())
                last_state[agent_name] = mem_last_state
                decoder_output[agent_name] = mem_decoder_output
                sequential_inputs.append(mem_decoder_output)

                decoder_outputs.append(mem_decoder_output)
                last_states.append(mem_last_state)
        # Topics:
        if len(self.sequential_topics) > 0:
            sequential_inputs.append(memory_dict['topic_embed'].unsqueeze(0))
        if self.sequential_state_manager is None:
            global_state = last_states[0]
            for idx in range(1, len(last_states)):
                global_state = global_state + last_states[idx]
            global_state = global_state / len(last_states)
        else:
            _, global_state = self.sequential_state_manager(torch.cat(sequential_inputs, -1), last_global_state)
        return decoder_output, last_state, global_state

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
        if not isinstance(last_state, tuple):
            last_state = last_state
            global_state = last_state + 0.0
        else:
            last_state, global_state = last_state
        decoder_output, last_state, global_state = self._update_decoder_state(last_state, global_state,
                                                                              current_input, memory_dict)
        tmp = self.sequential_state_down_scale(global_state[-1].squeeze(0))
        vocab_in = torch.cat([tmp, current_input], -1)

        if self.use_vocab_agent:
            gen_vocab_logits = self.vocab_predictor(vocab_in)
            gen_vocab_probs = torch.softmax(gen_vocab_logits, -1)
            padding_size = self.extend_vocab_size
            if padding_size > 0:
                gen_vocab_probs = torch.cat([gen_vocab_probs, torch.zeros([batch_size, padding_size],
                                                                          device=gen_vocab_probs.device)], dim=-1)
            else:
                gen_vocab_probs = gen_vocab_probs

            # 拷贝概率
            local_probs = [gen_vocab_probs]
        else:
            local_probs = []

        for agent_name in self.memory_agents.keys():
            agent = self.memory_agents[agent_name]
            if agent_name in self.stop_agent_flags:
                continue
            agent_in = decoder_output[agent_name].squeeze(0)
            local_copy_prob = agent.generate_vocab(vocab_in, agent_in, dynamic_projections,
                                                   pad_to_max=self.generation_mode_fusion_mode == 'dynamic_mode_fusion')
            local_probs += local_copy_prob

        # 合并
        if self.generation_mode_fusion_mode == 'dynamic_mode_fusion':
            token_probs = torch.cat(local_probs, -1)
        else:
            global_gates = torch.softmax(self.global_mode_selector(vocab_in), -1)
            res = 0.0
            assert global_gates.size()[-1] == len(local_probs)
            for idx, local_prob in enumerate(local_probs):
                res = res + local_prob * global_gates[:, idx:idx + 1]
            token_probs = res

        return torch.log(token_probs + 1e-20), (last_state, global_state)
