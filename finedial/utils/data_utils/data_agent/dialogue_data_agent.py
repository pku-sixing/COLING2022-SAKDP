import copy

from torchtext.data import Field

from finedial.ops.data_unit.Fields.BatchDynamicVocabField import BDVField
from finedial.utils.data_utils.input_helper import load_vocab, load_vocab_from_list
from finedial.utils.data_utils.param_helper import ParamDict
from finedial.utils.data_utils.tokenization.tokenization_helper import get_tokenizer
from finedial.utils.logging.logger_helper import logger
from finedial.utils.pretrained_utils import pretrained_helper

sp_tokens = ['<unk>', '<pad>', '<ssos>', '<seos>', '<sos>', '<eos>']
bert_sp_tokens = ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '<sos>', '<eos>']
sp_token_dict = {
    'unk': '<unk>',
    'pad': '<pad>',
    'ssos': '<ssos>',
    'seos': '<seos>',
    'sos': '<sos>',
    'eos': '<eos>',
}
bert_sp_token_dict = {
    'unk': '[UNK]',
    'pad': '[PAD]',
    'ssos': '[CLS]',
    'seos': '[SEP]',
    'sos': '<sos>',
    'eos': '<eos>',
}
SRC_SUFFIX = 'src'
TGT_SUFFIX = 'tgt'


class DialogueHelper:

    def __init__(self, params):
        self.max_line = params.dataset['max_line']
        self.has_src = params.dataset.get('has_src', True)
        self.tgt_vocab = None
        self.pretrained_encode_mode = params.dataset.get('pretrained_encode_mode', None)
        self.full_pretrained_vocab = params.dataset.get('full_pretrained_vocab', False)
        self.pair_wise_input_mode = params.dataset.get('pair_wise_input_mode', False)
        self.pretrained_config = False
        self.re_tokenizer = None
        if self.full_pretrained_vocab or self.pretrained_encode_mode == 'src_bert':
            self.sp_tokens = bert_sp_tokens
            self.sp_token_dict = ParamDict(bert_sp_token_dict)
            self.pretrained_config = params.pretrained_lm_configs['query']
            tokenizer, pretrained_vocab = pretrained_helper.get_tokenizer_and_vocab(self.pretrained_config)
            self.pretrained_vocab = pretrained_vocab

            sp_token_path = params.dataset.get('sp_token_vocab_path', None)
            if sp_token_path is not None:
                with open(sp_token_path, 'r+',encoding='utf-8') as fin:
                    additional_sp_tokens = fin.readlines()
                    additional_sp_tokens = [x.strip('\r\n') for x in additional_sp_tokens]
                logger.info('[Additional SPTokens] %s' % ' '.join(additional_sp_tokens))
                self.sp_tokens += additional_sp_tokens

            if self.full_pretrained_vocab:
                logger.info('[DATASET] The query is set to BERT mode, and the response follows the tokenizer'
                            ' transformers==%s' % self.pretrained_config.version)
            else:
                logger.info('[DATASET] The query input is set to BERT mode, transformers==%s' %
                            self.pretrained_config.version)

            if self.pretrained_config.tokenize == 'none':
                pass
            elif self.pretrained_config.tokenize == 'from_std_re_split':
                self.re_tokenizer = lambda x: tokenizer.tokenize(x)
            else:
                raise NotImplementedError()
        elif self.pretrained_encode_mode is None:
            self.sp_tokens = sp_tokens
            self.sp_token_dict = ParamDict(sp_token_dict)

        if not self.has_src:
            logger.info('[DATASET] The dialogue data agent will ignores the query')

        self.batch_size = params.dataset['batch_size']
        self.infer_batch_size = params.infer.get('batch_size', self.batch_size)
        self.truncate_first = params.infer.get('truncate_first', self.batch_size)
        self.max_tgt_len = params.dataset['max_tgt_len']
        assert self.max_tgt_len > 0
        self.tgt_suffix = params.dataset.get('response_suffix', TGT_SUFFIX)

        if self.has_src:
            self.max_src_len = params.dataset['max_src_len']
            assert self.max_src_len > 0
            self.src_suffix = params.dataset.get('query_suffix', SRC_SUFFIX)

        self.batch_first = params.get('mp_gpu', 0) > 1
        if self.has_src and 'attentive_memory' in params:
            self.copy_query = 'query' in params.attentive_memory and params.attentive_memory.query.copy_mode is not False
        else:
            self.copy_query = False

        if 'stepwise_decoder' in params.__dict__.keys():
            self.generation_mode_fusion_mode = params.stepwise_decoder['generation_mode_fusion_mode']
        else:
            self.generation_mode_fusion_mode = 'none'

        if self.has_src:
            logger.info('[DATASET] SRC_SUFFIX=%s, TGT_SUFFIX=%s' % (self.src_suffix, self.tgt_suffix))
        else:
            logger.info('[DATASET] TGT_SUFFIX=%s' % (self.tgt_suffix))

    def get_special_tokens(self):
        return self.sp_tokens

    def get_special_token_dict(self):
        res = {}
        for key, value in self.sp_token_dict.items():
            res[key] = self.tgt_vocab.stoi[value]
        return ParamDict(res)

    def get_local_vocab_size(self, params):
        if not self.has_src:
            return 0
        if 'attentive_memory' in params and \
                'query' in params.attentive_memory and params.attentive_memory.query.copy_mode is not False:
            return params.dataset.max_src_len + 2
        else:
            return 0

    def process_examples(self, my_examples, field_dict, batch_mode=True):
        # 裁剪一部分数据
        if self.max_line != -1:
            for idx in my_examples.keys():
                my_examples[idx] = my_examples[idx][0:self.max_line]

        if self.pretrained_encode_mode == 'src_bert' or self.full_pretrained_vocab:
            if self.pretrained_config.tokenize == 'from_std_re_split':
                my_examples['src'] = [' '.join(self.re_tokenizer(x)) for x in my_examples['src']]
        if self.full_pretrained_vocab:
            if self.pretrained_config.tokenize == 'from_std_re_split':
                my_examples['tgt'] = [' '.join(self.re_tokenizer(x)) for x in my_examples['tgt']]

        if not batch_mode:
            assert len(my_examples['tgt']) % self.batch_size != 1 and \
                   len(my_examples['tgt']) % self.infer_batch_size != 1, \
                'batch = 1 is not supported in batch_mode'

        if batch_mode:
            if self.has_src:
                logger.info('[DATASET] Maximum source dialogue sequence length is set to %d' % self.max_src_len)
            logger.info('[DATASET] Maximum target dialogue sequence length is set to %d' % self.max_tgt_len)

        if self.truncate_first:
            if self.has_src:
                my_examples['src'] = [' '.join(x.strip('\r\n').split()[0:self.max_src_len]) for x in my_examples['src']]
            my_examples['tgt'] = [' '.join(x.strip('\r\n').split()[0:self.max_tgt_len]) for x in my_examples['tgt']]
        else:
            if self.has_src:
                my_examples['src'] = [' '.join(x.strip('\r\n').split()[-self.max_src_len:]) for x in my_examples['src']]
            my_examples['tgt'] = [' '.join(x.strip('\r\n').split()[-self.max_tgt_len:]) for x in my_examples['tgt']]

        if 'dmc_bdv_query' in field_dict:
            assert self.has_src
            logger.info('[DATASET] Copy dmc_query from src, lines=%d' % (len(my_examples['src'])))
            my_examples['dmc_bdv_query'] = copy.deepcopy(my_examples['src'])

        # 创建适用于拷贝词表的部分
        if self.copy_query and self.has_src:
            local_vocabs = []
            if self.generation_mode_fusion_mode == 'dynamic_mode_fusion':
                for src_example in my_examples['src']:
                    local_vocabs.append(1)
                pass
            elif self.generation_mode_fusion_mode == 'dynamic_vocab':
                known_words = set(field_dict['tgt'].vocab.stoi.keys()) | set(self.get_special_tokens())
                for src_example in my_examples['src']:
                    local_vocab = set(src_example.split()) - known_words
                    local_vocabs.append(local_vocab)

            return my_examples, {'query': (self.max_src_len + 2, local_vocabs)}
        else:
            return my_examples, {}

    def load_examples(self, example_path, field_dict):
        my_examples = dict()
        # SRC
        if self.has_src:
            with open(example_path + '.' + self.src_suffix, 'r', encoding='utf-8') as fin:
                src = [x.strip('\n') for x in fin.readlines()]
            logger.info('[DATASET] Loaded %s.src(%s), lines=%d' % (example_path, self.src_suffix, len(src)))
            my_examples['src'] = src
        # TGT
        with open(example_path + '.' + self.tgt_suffix, 'r', encoding='utf-8') as fin:
            tgt = fin.readlines()
            logger.info('[DATASET] Loaded %s.tgt(%s), lines=%d' % (example_path, self.tgt_suffix, len(tgt)))
            my_examples['tgt'] = tgt

        return self.process_examples(my_examples, field_dict)

    def create_fields(self, params, additional_params, tokenizer='default'):
        # Dynamic_vocab
        build_dynamic_field = additional_params['generation_mode_fusion_mode'] == 'dynamic_vocab'

        # 开始加载数据集
        if self.has_src:
            src_vocab_path = params.dataset['src_vocab_path']
        tgt_vocab_path = params.dataset['tgt_vocab_path']
        share_src_tgt_vocab = params.dataset['share_src_tgt_vocab']

        tokenize = get_tokenizer(tokenizer)
        if self.has_src:
            src_field = Field(tokenize=tokenize, include_lengths=True, batch_first=self.batch_first,
                              init_token=self.sp_token_dict.ssos, eos_token=self.sp_token_dict.seos,
                              unk_token=self.sp_token_dict.unk, pad_token=self.sp_token_dict.pad)

        if self.pair_wise_input_mode:
            tgt_field = Field(tokenize=tokenize, include_lengths=True, batch_first=self.batch_first,
                              init_token=self.sp_token_dict.ssos, eos_token=self.sp_token_dict.seos,
                              unk_token=self.sp_token_dict.unk, pad_token=self.sp_token_dict.pad)
        else:
            tgt_field = Field(tokenize=tokenize, include_lengths=True, batch_first=self.batch_first,
                              init_token=self.sp_token_dict.sos, eos_token=self.sp_token_dict.eos,
                              unk_token=self.sp_token_dict.unk, pad_token=self.sp_token_dict.pad)

        # 创建词表
        if self.full_pretrained_vocab:
            logger.info('[VOCAB] Loading src-dialogue vocab the pretrained-LM:')
            tgt_vocab = load_vocab_from_list(self.pretrained_vocab,
                                             sp_tokens=self.sp_tokens)
            assert len(tgt_vocab.stoi) == len(self.pretrained_vocab), '%s-%s' % (len(tgt_vocab.stoi), len(self.pretrained_vocab))
            tgt_field.vocab = tgt_vocab
        else:
            logger.info('[VOCAB] Loading tgt-dialogue vocab from: %s' % tgt_vocab_path)
            tgt_vocab = load_vocab(tgt_vocab_path, self.get_special_tokens(),
                                   placeholder_tokens=additional_params['max_copy_vocab_size']
                                   , reset_unk=self.sp_token_dict.unk)
            tgt_field.vocab = tgt_vocab

        self.tgt_vocab = tgt_vocab

        if self.has_src:
            if share_src_tgt_vocab:
                src_field.vocab = tgt_vocab
            else:
                if self.pretrained_encode_mode == 'src_bert' or self.full_pretrained_vocab:
                    logger.info('[VOCAB] Loading src-dialogue vocab the pretrained-LM:')
                    src_vocab = load_vocab_from_list(self.pretrained_vocab,
                                                     sp_tokens=self.sp_tokens)
                    assert len(src_vocab.stoi) == len(self.pretrained_vocab)
                    src_field.vocab = src_vocab
                else:
                    logger.info('[VOCAB] Loading src-dialogue vocab from: %s' % src_vocab_path)
                    src_vocab = load_vocab(src_vocab_path, self.get_special_tokens())
                    src_field.vocab = src_vocab

        if self.has_src:
            result = [('src', src_field), ('tgt', tgt_field)]
            if build_dynamic_field:
                dmc_query_field = BDVField(tokenize=tokenize, include_lengths=False, batch_first=self.batch_first,
                                           init_token=self.sp_token_dict.ssos, eos_token=self.sp_token_dict.seos,
                                           unk_token=self.sp_token_dict.unk, pad_token=self.sp_token_dict.pad,
                                           dynamic_vocab=True)
                dmc_query_field.vocab = tgt_vocab
                result.append(('dmc_bdv_query', dmc_query_field))
        else:
            result = [('tgt', tgt_field)]

        return result
