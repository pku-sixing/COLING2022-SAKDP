import copy

import torch
from torchtext.data import Field
import transformers

from finedial.utils.data_utils.input_helper import load_vocab, load_vocab_from_list
from finedial.utils.data_utils.param_helper import ParamDict
from finedial.utils.data_utils.tokenization.tokenization_helper import get_tokenizer
from finedial.utils.logging.logger_helper import logger
from finedial.ops.data_unit.Fields.BatchDynamicVocabField import BDVField
from transformers import BertTokenizer

from finedial.utils.pretrained_utils import pretrained_helper

sp_tokens = ['<unk>', '<pad>', '<ssos>', '<seos>']
bert_tokens = ['[UNK]', '[PAD]', '[CLS]', '[SEP]']

sp_token_dict = {
    'unk': '<unk>',
    'pad': '<pad>',
    'sos': '<ssos>',
    'eos': '<seos>',
}
bert_sp_token_dict = {
    'unk': '[UNK]',
    'pad': '[PAD]',
    'sos': '[CLS]',
    'eos': '[SEP]',
}

TEXT_SUFFIX = 'text'


class TextReferenceHelper:

    def __init__(self, params, reference_params):
        self.reference_params = reference_params
        self.suffix = reference_params.dataset.get('suffix', TEXT_SUFFIX)
        self.batch_first = params.get('mp_gpu', 0) > 1
        self.max_line = params.dataset['max_line']
        if 'stepwise_decoder' in params.__dict__.keys():
            self.generation_mode_fusion_mode = params.stepwise_decoder['generation_mode_fusion_mode']
        else:
            self.generation_mode_fusion_mode = 'none'

        self.pretrained_encode_mode = reference_params.dataset.get('pretrained_encode_mode', None)
        self.full_pretrained_vocab = reference_params.dataset.get('full_pretrained_vocab', False)
        self.pretrained_config_name = "text_bert"
        self.pretrained_config = False
        self.re_tokenizer = None

        if self.pretrained_encode_mode == 'bert' or self.full_pretrained_vocab:
            self.sp_tokens = bert_tokens
            self.sp_token_dict = ParamDict(bert_sp_token_dict)
            self.pretrained_config = reference_params.pretrained_lm_configs[self.pretrained_config_name]

            tokenizer, pretrained_vocab = pretrained_helper.get_tokenizer_and_vocab(self.pretrained_config)
            self.pretrained_vocab = pretrained_vocab
            # 目前暂时要确保这个一致
            logger.info('[DATASET] The TEXT input is set to BERT mode, transformers==%s' %
                        self.pretrained_config.version)
            assert self.pretrained_config.version == transformers.__version__, '%s-%s' % (
                self.pretrained_config.version, transformers.__version__
            )
            # 是否需要额外分词，文件默认输入时空格分词
            if self.pretrained_config.tokenize == 'none':
                pass
            elif self.pretrained_config.tokenize == 'from_std_re_split':
                self.re_tokenizer = lambda x: tokenizer.tokenize(x)
            else:
                raise NotImplementedError()
        elif self.pretrained_encode_mode is None:
            self.sp_tokens = sp_tokens
            self.sp_token_dict = ParamDict(sp_token_dict)
        else:
            raise NotImplementedError()


    def get_special_tokens(self):
        return self.sp_tokens

    def get_local_vocab_size(self, reference_name):
        # TODO 增加一条Summary
        if 'attentive_memory' in self.reference_params and \
                (self.reference_params.attentive_memory.copy_mode is not False
                  or self.reference_params.get('module_mode') == 'placeholder'):
            return self.reference_params.dataset.max_len + 1
        else:
            return 0

    def load_examples(self, example_path, field_dict):
        reference_params = self.reference_params
        max_line = self.max_line
        max_len = reference_params.dataset['max_len']
        reference_name = reference_params['name']
        assert max_len > 0
        my_examples = dict()

        # SRC
        with open(example_path + '.' + self.suffix, 'r', encoding='utf-8') as fin:
            src = [x.strip('\n') for x in fin.readlines()]
        logger.info('[DATASET] Loaded reference %s form %s.text, lines=%d' % (reference_name, example_path, len(src)))
        my_examples[reference_name] = src

        # 裁剪一部分数据
        if max_line != -1:
            for idx in my_examples.keys():
                my_examples[idx] = my_examples[idx][0:max_line]

        # 利用Bert重新分词
        if self.pretrained_encode_mode == 'bert' or self.full_pretrained_vocab:
            if self.pretrained_config.tokenize == 'from_std_re_split':
                my_examples[reference_name] = [' '.join(self.re_tokenizer(x)) for x in my_examples[reference_name]]


        logger.info('[DATASET] Maximum %s sequence length is set to %d' % (reference_name, max_len))
        for idx in range(len(my_examples[reference_name])):
            x = my_examples[reference_name][idx]
            my_examples[reference_name][idx] = ' '.join(x.strip('\r\n').split()[0:max_len])

        dmc_name = 'dmc_bdv_' + reference_name
        if dmc_name in field_dict:
            logger.info('[DATASET] Copy %s from %s, lines=%d' % (dmc_name, reference_name, len(src)))
            my_examples[dmc_name] = copy.deepcopy(my_examples[reference_name])

        # 创建适用于拷贝词表的部分
        logger.info('[DATASET] create extend vocabs for %s' % reference_name)
        local_vocabs = None
        if 'attentive_memory' in reference_params and reference_params.attentive_memory.copy_mode is not False:
            local_vocabs = []
            generation_mode_fusion_mode = self.generation_mode_fusion_mode
            if generation_mode_fusion_mode == 'dynamic_mode_fusion':
                for src_example in my_examples[reference_name]:
                    local_vocabs.append(1)
            elif generation_mode_fusion_mode == 'dynamic_vocab':
                known_words = set(field_dict['tgt'].vocab.stoi.keys()) | set(self.get_special_tokens())
                for src_example in my_examples[reference_name]:
                    local_vocab = set(src_example.split()) - known_words
                    local_vocabs.append(local_vocab)

            return my_examples, {reference_name: (reference_params.dataset.max_len + 2, local_vocabs)}
        else:
            return my_examples, {}

    def create_fields(self, params, additional_params, tokenizer='default'):

        reference_params = self.reference_params
        reference_name = reference_params['name']

        # dynamic_vocab
        build_dynamic_field = additional_params['generation_mode_fusion_mode'] == 'dynamic_vocab'

        # 开始加载数据集
        vocab_mode = reference_params.dataset['vocab_mode']
        src_vocab_path = reference_params.dataset['vocab_path']

        tokenize = get_tokenizer(tokenizer)
        src_field = Field(tokenize=tokenize, include_lengths=True,
                          init_token=self.sp_token_dict.sos, eos_token=self.sp_token_dict.eos,
                          unk_token=self.sp_token_dict.unk, pad_token=self.sp_token_dict.pad
                          , batch_first=self.batch_first)

        # 创建词表
        if vocab_mode == 'separate':
            if self.pretrained_encode_mode == 'bert' or self.full_pretrained_vocab:
                logger.info('[VOCAB] Loading src-dialogue vocab the pretrained-LM:')
                src_vocab = load_vocab_from_list(self.pretrained_vocab)
                assert len(src_vocab.stoi) == len(self.pretrained_vocab)
                src_field.vocab = src_vocab
            else:
                logger.info('[VOCAB] Loading %s-TEXT vocab from: %s' % (reference_name, src_vocab_path))
                src_vocab = load_vocab(src_vocab_path, self.get_special_tokens())
                src_field.vocab = src_vocab
        elif vocab_mode == 'share':
            logger.info('[VOCAB] Do not load  %s-TEXT vocab' % reference_name)
            field_dict = additional_params['field_dict']
            if 'src' in field_dict and not (self.pretrained_encode_mode == 'bert' or self.full_pretrained_vocab):
                src_field.vocab = field_dict['src'].vocab
            else:
                src_field.vocab = field_dict['tgt'].vocab
        elif vocab_mode == 'share_tgt':
            logger.info('[VOCAB] Do not load  %s-TEXT vocab' % reference_name)
            field_dict = additional_params['field_dict']
            src_field.vocab = field_dict['tgt'].vocab
        else:
            raise NotImplementedError()

        result = [
            (reference_name, src_field),
        ]

        if build_dynamic_field:
            dmc_src_field = BDVField(tokenize=tokenize, include_lengths=False,
                                     init_token=self.sp_token_dict.sos, eos_token=self.sp_token_dict.eos,
                                     unk_token=self.sp_token_dict.unk, pad_token=self.sp_token_dict.pad,
                                     batch_first=self.batch_first, dynamic_vocab=True)
            field_dict = additional_params['field_dict']
            dmc_src_field.vocab = field_dict['tgt'].vocab
            result.append(('dmc_bdv_' + reference_name, dmc_src_field))

        return result
