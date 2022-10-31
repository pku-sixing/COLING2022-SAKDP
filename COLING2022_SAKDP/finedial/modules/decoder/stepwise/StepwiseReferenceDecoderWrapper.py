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


class StepwiseReferenceDecoder(StepwiseDecoderBase):

    def __init__(self, params, memory_agents={}, reference_agents={}):
        super(StepwiseReferenceDecoder, self).__init__(params)
        self.memory_agents = memory_agents
        self.reference_agents = reference_agents

        self.infuse_global_state = params.get('infuse_global_state', False)
        self.fusion_hidden_size = params.get('fusion_hidden_size', 1024)
        self.compute_channel_loss = params.get('compute_channel_loss', False)
        self.use_global_channel_gate = params.get('global_channel_gate', False)
        self.disable_global_channel_gate_in_infer = params.get('disable_global_channel_gate_in_infer', False)

        if self.use_global_channel_gate is not False:
            assert params.reference_fusion_mode == 'sequential'

        # 配置Memory Agent
        self.generation_mode_num = 0
        self.memory_readout_dim = 0
        self.decoder_output_size = 0
        self.attention_output_size = []
        self.memory_output_dim = 0
        self.memory_agent_order = []
        self.decoder_output_size = []
        self.summary_size = 0
        for agent_name, agent in reference_agents.items():
            if agent.has_decoder:
                self.decoder_output_size.append(agent.decoder_output_size)
                self.generation_mode_num += 1
                self.memory_agent_order.append(agent_name)
            if agent.has_attention:
                self.attention_output_size.append(agent.memory_agent.get_output_dim())

        if self.generation_mode_num > 1:
            self.input_size = params.input_size
            self.reference_fusion_mode = params.reference_fusion_mode
            if self.reference_fusion_mode == 'none':
                pass
            elif self.reference_fusion_mode == 'sequential_stepwise':
                # Input: Hidden State + All Attentions +  Last Output
                input_size = sum(self.decoder_output_size) + sum(self.attention_output_size) + self.input_size
                fusion_hidden_size = self.fusion_hidden_size
                self.sequential_state_manager = rnn_helper.create_rnn_network('gru', input_size,
                                                                              fusion_hidden_size, 1,
                                                                              bias=True, dropout=params.dropout)
                self.stepwise_channel_gate = dict()
                for idx, channel_name in enumerate(self.memory_agent_order):
                    self.stepwise_channel_gate[channel_name] = nn.Sequential(
                        nn.Linear(self.decoder_output_size[idx] + self.attention_output_size[idx] +
                                  fusion_hidden_size + self.input_size, fusion_hidden_size // 2, bias=True),
                        torch.nn.Tanh(),
                        nn.Linear(fusion_hidden_size // 2, 1, bias=True),
                    )
                self.stepwise_channel_gate = nn.ModuleDict(self.stepwise_channel_gate)
            elif self.reference_fusion_mode == 'sequential':
                # Input: Hidden State + All Attentions +  Last Output
                input_size = sum(self.decoder_output_size) + sum(self.attention_output_size)
                fusion_hidden_size = self.fusion_hidden_size
                self.sequential_state_manager = rnn_helper.create_rnn_network('gru', input_size,
                                                                              fusion_hidden_size, 1,
                                                                              bias=True, dropout=params.dropout)

                input_size = sum(self.decoder_output_size) + sum(self.attention_output_size) + self.input_size
                self.stepwise_channel_gate = nn.Sequential(
                    nn.Linear(input_size, fusion_hidden_size, bias=True),
                    torch.nn.Tanh(),
                    nn.Linear(fusion_hidden_size, len(self.memory_agent_order), bias=True),
                    torch.nn.Softmax(dim=-1)
                )

                if self.use_global_channel_gate == 'softmax':
                    self.global_channel_gate = nn.Sequential(
                        nn.Linear(sum(self.attention_output_size), fusion_hidden_size, bias=True),
                        torch.nn.Tanh(),
                        nn.Linear(fusion_hidden_size, len(self.memory_agent_order), bias=True),
                        torch.nn.Softmax(dim=-1)
                    )
                elif self.use_global_channel_gate == 'sigmoid':
                    self.global_channel_gate = nn.Sequential(
                        nn.Linear(sum(self.attention_output_size), fusion_hidden_size, bias=True),
                        torch.nn.Tanh(),
                        nn.Linear(fusion_hidden_size, len(self.memory_agent_order), bias=True),
                        torch.nn.Sigmoid()
                    )



            else:
                raise NotImplementedError()

    def forward(self, decoder_state):
        """
            Inputs:
                last_state:
                    state: 当前的decoder状态信息
                    current_input: [batch_size, input_dim]当前的decoder输入
                    dynamic_vocab_projections: 投影到动态词表上
        """

        assert isinstance(decoder_state, StepwiseDecoderState)
        last_state_dict = decoder_state.state
        assert isinstance(last_state_dict, dict), 'last decoder state should be a dict'
        current_input = decoder_state.input

        gen_prob_dists = []
        gen_mode_names = []
        attention_readouts = []
        agent_outputs = []
        channel_prob_dists = []
        for agent_name, agent in self.reference_agents.items():
            if agent.has_decoder is False:
                continue
            last_state = last_state_dict[agent_name]
            if not self.infuse_global_state:
                last_state, gen_vocab_dist, attention_readout = agent.decode_step(last_state, current_input)
            else:
                _layer, _batch, _dim = last_state_dict['context'].size()
                global_state = last_state_dict.get('sequential_manager',
                                                   torch.zeros([_layer, _batch, self.fusion_hidden_size],
                                                               device=last_state_dict['context'].device))
                new_input = torch.cat([global_state.squeeze(0), current_input], -1)
                last_state, gen_vocab_dist, attention_readout = agent.decode_step(last_state, new_input)
            last_state_dict[agent_name] = last_state
            gen_prob_dists.append(gen_vocab_dist.unsqueeze(0))
            gen_mode_names.append(agent_name)
            attention_readouts.append(attention_readout)
            agent_outputs.append(last_state[-1])

        assert len(gen_mode_names) == len(gen_prob_dists)
        assert len(gen_mode_names) == self.generation_mode_num

        if self.generation_mode_num > 1:
            if self.reference_fusion_mode == 'stepwise':
                inputs = agent_outputs + attention_readouts + [current_input]
                relevance_gates = self.stepwise_channel_gate(torch.cat(inputs, -1))
            elif self.reference_fusion_mode == 'sequential_stepwise':
                inputs = agent_outputs + attention_readouts + [current_input]
                sequential_state = last_state_dict.get('sequential_manager', None)
                inputs = torch.cat(inputs, -1)
                inputs = inputs.unsqueeze(0)
                sequential_output, last_sequential_state = self.sequential_state_manager(inputs, sequential_state)
                last_state_dict['sequential_manager'] = last_sequential_state

                relevance_gates = []
                for idx, agent_name in enumerate(self.memory_agent_order):
                    reference_gate_input = torch.cat([current_input, agent_outputs[idx], attention_readouts[idx],
                                                      sequential_output.squeeze(0)], -1)
                    relevance_gate = self.stepwise_channel_gate[agent_name](
                        reference_gate_input)
                    relevance_gates.append(relevance_gate)
                relevance_gates = torch.cat(relevance_gates, -1)
                relevance_gates = torch.softmax(relevance_gates, -1)
            elif self.reference_fusion_mode == 'sequential':
                # TODO 没有用到Sequential State
                inputs = agent_outputs + attention_readouts
                if self.infuse_global_state:
                    sequential_state = last_state_dict.get('sequential_manager', None)
                    inputs = torch.cat(inputs, -1)
                    inputs = inputs.unsqueeze(0)
                    sequential_output, last_sequential_state = self.sequential_state_manager(inputs, sequential_state)
                    last_state_dict['sequential_manager'] = last_sequential_state
                else:
                    assert 'sequential_manager' not in last_state_dict

                reference_gate_input = torch.cat([current_input] + agent_outputs + attention_readouts, -1)
                relevance_gates = self.stepwise_channel_gate(
                    reference_gate_input)

                if self.use_global_channel_gate is not False and self.disable_global_channel_gate_in_infer is False:
                    if 'global_relevance_gates' not in last_state_dict:
                        global_relevance_gates = self.global_channel_gate(torch.cat(attention_readouts, -1))
                        last_state_dict['global_relevance_gates'] = global_relevance_gates.unsqueeze(0)
                    global_relevance_gates = last_state_dict['global_relevance_gates'].squeeze(0)
                    relevance_gates = relevance_gates * global_relevance_gates
                    relevance_gates = relevance_gates / relevance_gates.sum(-1, keepdims=True)

            if self.reference_fusion_mode != 'none':
                for gen_id in range(len(gen_mode_names)):
                    relevance_gate = relevance_gates[:, gen_id]
                    relevance_gate = relevance_gate.unsqueeze(0).unsqueeze(-1)
                    channel_prob_dists.append(gen_prob_dists[gen_id])
                    gen_prob_dists[gen_id] = gen_prob_dists[gen_id] * relevance_gate
            else:
                for gen_id in range(len(gen_mode_names)):
                    relevance_gate = 1.0 / len(gen_mode_names)
                    channel_prob_dists.append(gen_prob_dists[gen_id])
                    gen_prob_dists[gen_id] = gen_prob_dists[gen_id] * relevance_gate
        else:
            channel_prob_dists = gen_prob_dists

        decoder_state.token_prob_dist = gen_prob_dists
        if self.compute_channel_loss:
            decoder_state.channel_token_prob_dist = channel_prob_dists
        decoder_state.token_mode_order = gen_mode_names
        decoder_state.state = last_state_dict
        return decoder_state
