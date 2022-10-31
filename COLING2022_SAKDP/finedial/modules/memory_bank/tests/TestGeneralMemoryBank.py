import pprint
import unittest

import torch

from finedial.modules.memory_bank.GeneralMemoryBank import GeneralMemoryBankAgent
from finedial.ops.paramters import parameter_helper


class MyTestCase(unittest.TestCase):
    def test_something(self):
        params_configs = {
            'attention_type': ['mlp', 'dot', 'general'],
            'query_dim': [1, 11, 32],
            'value_dim': [1, 11, 32],
            'key_dim': [1, 11, 32],
            'hidden_dim': [1, 6],
            'copy_mode':[True, False],
            'projection_flags': [None,
                                 {'query':{'in_dim': 11, 'out_dim':20, 'activation':'tanh'},
                                  'key':{'in_dim': 11, 'out_dim':20},
                                  'value':{'in_dim': 11, 'out_dim':20}
                                  }
                                 ]

        }
        params_list = parameter_helper.enumerate_params(params_configs)
        for test_case, params in enumerate(params_list):
            print('test progress', test_case, '/', len(params_list))

            if params['projection_flags'] is None:
                if params['query_dim'] != params['key_dim'] and params['attention_type'] == 'dot':
                    continue
            else:
                if 'query' in params['projection_flags']:
                    params['projection_flags']['query']['in_dim'] = params['query_dim']
                    params['projection_flags']['query']['out_dim'] = params['hidden_dim']
                if 'value' in params['projection_flags']:
                    params['projection_flags']['value']['in_dim'] = params['value_dim']
                    params['projection_flags']['value']['out_dim'] = params['hidden_dim']
                if 'key' in params['projection_flags']:
                    params['projection_flags']['key']['in_dim'] = params['key_dim']
                    params['projection_flags']['key']['out_dim'] = params['hidden_dim']
            pprint.pprint(params)
            general_memory_agent = GeneralMemoryBankAgent(params)
            print(general_memory_agent)

            seq_len_candidates = [10, 11, 1]
            batch_candidates = [10, 11, 1]
            for seq_len in seq_len_candidates:
                for batch in batch_candidates:
                    key_dim = params['key_dim']
                    value_dim = params['value_dim']
                    query_dim = params['query_dim']
                    memory_bank = torch.rand([batch, seq_len, key_dim])
                    memory_bank_value = torch.rand([batch, seq_len, value_dim])
                    lengths = torch.ones([batch]) * seq_len
                    general_memory_agent.init_memory_bank(memory_bank, memory_bank_value, lengths)

                    query = torch.rand([batch, query_dim])
                    vector, align_scores = general_memory_agent.access(query)
                    # [batch, bank_len]
                    a,  c = align_scores.size()
                    self.assertEqual(batch, a)
                    self.assertEqual(seq_len, c)


if __name__ == '__main__':
    unittest.main()
