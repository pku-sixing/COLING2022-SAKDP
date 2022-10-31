import unittest
import torch
import pprint
from finedial.modules.encoder.sequential.SequentialEncoder import SequentialEncoder
from finedial.ops.paramters import parameter_helper

class MyTestCase(unittest.TestCase):
    def test_sequential_encoder(self):
        """
        进测试是否能够完整运行
        """
        params_configs = {
            'sequential_encoder_type': ['rnn'],
            'rnn_type': ['gru', 'lstm'],
            'input_size': [1, 3, 5, 16, 32],
            'hidden_size': [1, 3, 5, 16, 32],
            'dropout': [0.0, 0.5, 1.0],
            'bidirectional': [True, False],
            'n_layers': [3, 2, 1],
            'bidirectional_rnn_outputs_aggregator': ['mlp', 'concat', 'sum']
        }
        params_list = parameter_helper.enumerate_params(params_configs)
        for test_case, params in enumerate(params_list):
            print('test progress', test_case, '/', len(params_list))
            pprint.pprint(params)
            encoder = SequentialEncoder(params)
            # inputs:
            # embed_seq: [seq_len, batch, embed_dim]
            # lengths: [batch]

            seq_len_candidates = [10, 11, 1]
            batch_candidates = [10, 11,  1]
            for seq_len in seq_len_candidates:
                for batch in batch_candidates:
                    embed_dim = params['input_size']
                    embed_seq = torch.rand([seq_len, batch, embed_dim])
                    lengths = torch.ones([batch]) * seq_len
                    outputs, last_state = encoder(embed_seq, lengths)
                    # outputs:  [seq_len, batch, hidden_size]
                    a, b, c = outputs.shape
                    self.assertEqual(seq_len, a)
                    self.assertEqual(batch, b)
                    if params['bidirectional'] and params['bidirectional_rnn_outputs_aggregator'] == 'concat':
                        self.assertEqual(params['hidden_size'] * 2, c)
                    else:
                        self.assertEqual(params['hidden_size'], c)
                    # last_state:  [num_layers , batch, hidden_size]
                    if isinstance(last_state, tuple):
                        last_state = last_state[0]
                    a, b, c = last_state.shape
                    self.assertEqual(params['n_layers'], a)
                    self.assertEqual(batch, b)
                    if params['bidirectional'] and params['bidirectional_rnn_outputs_aggregator'] == 'concat':
                        self.assertEqual(params['hidden_size'] * 2, c)
                    else:
                        self.assertEqual(params['hidden_size'], c)

            # self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
