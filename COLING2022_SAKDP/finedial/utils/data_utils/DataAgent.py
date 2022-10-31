from finedial.utils.data_utils import load_dataset


class DataAgent:
    def __init__(self, params):
        train_iter, val_iter, test_iter, field_dict, dialogue_sp_token_dicts = load_dataset.load_dataset(params)
        for batch in train_iter:
            print(batch)

if __name__ =='__main__':
    params = {
        'batch_size': 32,
        'tgt_vocab_size': 1000,
        'share_src_tgt_vocab': True,
        'cuda': False,
        'max_line': -1,
        'max_src_len': -1,
        'max_tgt_len': -1,
        'tgt_vocab_path': 'datasets/toy_seq2seq/vocab.txt',
        'src_vocab_path': 'datasets/toy_seq2seq/vocab.txt',
        'val_data_path_prefix': 'datasets/toy_seq2seq/test',
        'test_data_path_prefix': 'datasets/toy_seq2seq/test',
        'training_data_path_prefix': 'datasets/toy_seq2seq/test',
    }
    agent = DataAgent(params)