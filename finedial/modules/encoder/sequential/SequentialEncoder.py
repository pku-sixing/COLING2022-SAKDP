"""
用于处理Sequential 数据的编码
Time First
"""
import random

import torch
from torch import nn
from finedial.ops.network_unit import rnn_helper, transfomer_helper, projection_helper
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from finedial.ops.network_unit.transformer.position_ffn import PositionalEncoding


class SequentialEncoder(nn.Module):

    @classmethod
    def change_input_dim(cls, encoder_params, new_dim, field='sequential_encoder'):
        assert field in encoder_params.keys(), field
        if encoder_params[field].sequential_encoder_type == 'rnn':
            encoder_params[field].rnn_network.input_size = new_dim
        elif encoder_params[field].sequential_encoder_type == 'transformer':
            encoder_params[field].transformer_encoder_network.input_size = new_dim
        elif encoder_params[field].sequential_encoder_type == 'pretrained_encoder_adapter':
            pass
        else:
            raise NotImplementedError()

    def __init__(self, params):
        """
        params_configs = {
            'sequential_encoder_type': ['rnn'],
            'rnn_type': ['gru', 'lstm'],
            'input_size': [1, 3, 5, 16, 32],
            'hidden_size': [1, 3, 5, 16, 32],
            'dropout': [0.0, 0.5, 1.0],
            'bidirectional': [True, False],
            'n_layers': [3, 2, 1],
            'bidirectional_rnn_outputs_aggregator': ['concat', 'sum']
        }

        """
        super(SequentialEncoder, self).__init__()
        encoder_type = params['sequential_encoder_type']
        self.input_noise = params.get('input_noise', None)
        self.placeholder_encoder = False

        input_dropout = params.get('input_dropout', 0.0)
        if input_dropout > 0.0:
            self.input_dropout = nn.Dropout(input_dropout)
        else:
            self.input_dropout = None

        if encoder_type == 'rnn':
            assert 'rnn_network' in params
            self.encoder_type = 'rnn'
            rnn_params = params.rnn_network
            params.output_size = rnn_params['hidden_size']
            self.output_size = rnn_params['hidden_size']
            self.bidirectional = rnn_params['bidirectional']
            self.n_layers = rnn_params['n_layers']
            self.rnn_type = rnn_params['rnn_type']
            self.rnn = rnn_helper.create_rnn_network_from_params(rnn_params)
            if self.bidirectional is True:
                self.bidirectional_aggregator = BidirectionalRNNAggregator(rnn_params)
            else:
                self.bidirectional_aggregator = None
        elif encoder_type == 'transformer':
            assert 'transformer_encoder_network' in params
            self.encoder_type = 'transformer'
            trans_params = params.transformer_encoder_network
            params.output_size = trans_params['hidden_size']
            self.output_size = trans_params['hidden_size']
            # 如果需要距离不能感知，则需要将positional_encoding设置为False
            if trans_params.positional_encoding:
                assert trans_params.positional_encoding.type == 'std'
                self.positional_encoder = PositionalEncoding(trans_params.positional_encoding.dropout,
                                                             trans_params.hidden_size)
            else:
                self.positional_encoder = lambda x: x
            self.transformer = transfomer_helper.create_transformer_encoder_network_from_params(trans_params)
            if trans_params.input_size != trans_params.hidden_size:
                self.transformer_in_projection = projection_helper.create_projection_layer(
                    in_dim=trans_params.input_size, out_dim=trans_params.hidden_size, bias=False, activation='tanh')
            else:
                self.transformer_in_projection = lambda x:x
        elif encoder_type == 'none':
            self.encoder_type = 'none'
            self.placeholder_encoder = True
        elif encoder_type == 'pretrained_encoder_adapter':
            adapter_params = params.adapter_network
            params.output_size = adapter_params.output_size
            self.output_size = adapter_params.output_size
            self.encoder_type = 'pretrained_encoder_adapter'
        else:
            raise NotImplementedError()

    def forward(self, embed_seq, lengths, init_state=None, enforce_sorted=True):
        """
            Inputs:
                embed_seq : [seq_len, batch, embed_dim]
                lengths : [batch]
            Outputs:
                outputs:  [seq_len, batch, hidden_size]
                last_state:  [num_layers , batch, hidden_size] : GRU是最后一个状态，而Transformer把第一个位置当成是输出, 默认层数为1
        """
        # 是否增加噪音
        if self.input_dropout is not None:
            embed_seq = self.input_dropout(embed_seq)
        if self.input_noise is not None:
            if self.input_noise == 'default':
                if self.training:
                    mode_val = random.random()
                    if mode_val < 0.3:
                        noise = 0.0
                    elif mode_val < 0.2:
                        noise = torch.randn_like(embed_seq)
                    else:
                        rand_factor = random.random()
                        noise = torch.randn_like(embed_seq) * rand_factor
                    embed_seq += noise
            elif self.input_noise == 'default2':
                if self.training:
                    mode_val = random.random()
                    if mode_val < 0.3:
                        noise = 0.0
                    elif mode_val < 0.2:
                        noise = torch.randn_like(embed_seq) * 0.1
                    else:
                        rand_factor = random.random()
                        noise = torch.randn_like(embed_seq) * rand_factor * 0.1
                    embed_seq += noise
            else:
                raise NotImplementedError()
        if self.encoder_type == 'rnn':
            # Lengths data is wrapped inside a Tensor.
            # 多GPU下可能会出现问题
            raw_seq_len = embed_seq.size()[0]
            lengths_list = lengths.view(-1).tolist()
            embedding = pack(embed_seq, lengths_list, enforce_sorted=self.training & enforce_sorted)
            if self.rnn_type == 'lstm':
                # [IN] inputs-LSTM:  embedding (seq_len, batch, input_size)
                # [IN] last_state= h0/c0:(num_layers * num_directions, batch, hidden_size):
                # [OUT] outputs = (seq_len, batch, num_directions * hidden_size)
                # [out] last_state (num_layers * num_directions, batch, hidden_size)
                outputs, last_state = self.rnn(embedding, init_state)
                outputs = unpack(outputs)[0]
                if self.bidirectional:
                    # outputs
                    seq_len, batch, num_hidden = outputs.shape
                    outputs = outputs.view(seq_len, batch, 2, num_hidden // 2)
                    outputs = self.bidirectional_aggregator(outputs, direction_dim=2, data_dim=3)
                    tmp_res = []
                    for tmp_last_state in last_state:
                        # last_tate
                        num_layers_bi, batch, hidden_size = tmp_last_state.shape
                        num_layers = num_layers_bi // 2
                        tmp_last_state = tmp_last_state.view(num_layers, 2, batch, hidden_size)
                        tmp_last_state = self.bidirectional_aggregator(tmp_last_state, direction_dim=1, data_dim=3)
                        tmp_res.append(tmp_last_state)
                    last_state = tuple(tmp_res)
            elif self.rnn_type == 'gru':
                # [IN]:inputs-GRU:  embedding(seq_len, batch, input_size)
                # [IN]:inputs-init_state=h0: (num_layers * num_directions, batch, hidden_size)
                # [OT]:outputs:  (seq_len, batch, num_directions * hidden_size)
                # [OT]:last_state: (num_layers * num_directions, batch, hidden_size)
                outputs, last_state = self.rnn(embedding, init_state)
                outputs = unpack(outputs)[0]
                if self.bidirectional:
                    # outputs
                    seq_len, batch, num_hidden = outputs.shape
                    outputs = outputs.view(seq_len, batch, 2, num_hidden // 2)
                    # =>   (seq_len, batch, hidden_size)
                    outputs = self.bidirectional_aggregator(outputs, direction_dim=2, data_dim=3)
                    # last_tate
                    num_layers_bi, batch, hidden_size = last_state.shape
                    num_layers = num_layers_bi // 2
                    last_state = last_state.view(num_layers, 2, batch, hidden_size)
                    # => (num_layers , batch, hidden_size)
                    last_state = self.bidirectional_aggregator(last_state, direction_dim=1, data_dim=3)
            else:
                raise NotImplementedError()

            # new_seq_len
            new_seq_len, batch_size, dim = outputs.size()
            if new_seq_len < raw_seq_len:
                pad_num = raw_seq_len - new_seq_len
                padding = torch.zeros([pad_num, batch_size, dim], dtype=outputs.dtype, device=outputs.device,
                                      requires_grad=False)
                outputs = torch.cat([outputs, padding], dim=0)
            elif new_seq_len == raw_seq_len:
                pass
            else:
                raise NotImplementedError()
            return outputs, last_state
        elif self.encoder_type == 'transformer':
            trans_in = self.transformer_in_projection(embed_seq)
            trans_in = self.positional_encoder(trans_in)
            # 必须要指定长度，不然会失效
            src_mask, src_attn_mask = transfomer_helper.create_encoding_mask(lengths, trans_in.size()[0])
            outputs = self.transformer(trans_in, src_mask, src_attn_mask)
            # 把第一个位置当成是输出
            last_state = outputs[0:1]
            return outputs, last_state
        elif self.encoder_type == 'pretrained_encoder_adapter':
            # Bert Adapter 直接返回CLS的位置
            return embed_seq, embed_seq[0, :, :]
        else:
            raise NotImplementedError()


class BidirectionalRNNAggregator(nn.Module):

    def __init__(self, params):
        """
        如何将双向的GRU结果进行合并的:
            none
            sum 两个方向的相加
            concat 不合并， 默认就是concat 两个方向的拼接，维度翻倍
        """
        super(BidirectionalRNNAggregator, self).__init__()
        self.aggregator_method = params['bidirectional_rnn_outputs_aggregator']
        assert self.aggregator_method in {'none', 'sum', 'concat', 'mlp'}

        self.output_projection = None

        if self.aggregator_method == 'concat' or self.aggregator_method == 'mlp':
            self.aggregator = lambda x, y, d: torch.cat([x, y], dim=d)
            if self.aggregator_method == 'mlp':
                self.output_projection = nn.Sequential(
                    nn.Linear(
                        in_features=params['hidden_size'] * 2, out_features=params['hidden_size'],
                        bias=False),
                    nn.Tanh(),
                )
        elif self.aggregator_method == 'sum':
            self.aggregator = lambda x, y, d: (x + y)
        else:
            NotImplementedError()

    def forward(self, inputs, direction_dim, data_dim):
        """
        将输入沿着direction_dim的两个值进行合并
        """
        x_list = inputs.unbind(direction_dim)
        assert len(x_list) == 2

        if data_dim > direction_dim:
            data_dim -= 1

        aggregated_result = self.aggregator(x_list[0], x_list[1], data_dim)

        if self.output_projection is not None:
            aggregated_result = self.output_projection(aggregated_result)

        return aggregated_result
