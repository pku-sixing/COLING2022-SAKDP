import copy
from collections import defaultdict

import torch
from torchtext.data import RawField

from finedial.ops.data_unit.Fields.SubTextField import SubTextField
from finedial.utils.data_utils.input_helper import load_vocab
from finedial.utils.logging.logger_helper import logger

sp_tokens = ['<unk>', '<pad>', '<csos>', '<ceos>', '<w>','</w>', '<p>']
TEXT_SUFFIX = 'htext'


class SubTextReferenceHelper:

    def __init__(self, params, reference_params):
        self.reference_params = reference_params
        self.suffix = reference_params.dataset.get('suffix', TEXT_SUFFIX)
        self.batch_first = params.get('mp_gpu', 0) > 1
        self.max_len = reference_params.dataset.max_len
        self.context_split_notation = reference_params.dataset['context_split_notation']
        self.utterance_split_notation = reference_params.dataset['utterance_split_notation']
        self.max_line = params.dataset['max_line']
        self.generation_mode_fusion_mode = params.stepwise_decoder['generation_mode_fusion_mode']

    def get_special_tokens(self):
        return sp_tokens

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
        max_context_len = reference_params.dataset['max_context_len']
        max_utterance_len = reference_params.dataset['max_utterance_len']
        reference_name = reference_params['name']
        assert max_context_len > 0
        my_examples = dict()

        # SRC
        with open(example_path + '.' + self.suffix, 'r', encoding='utf-8') as fin:
            src = [x.strip('\n') for x in fin.readlines()]
        logger.info('[DATASET] Loaded reference %s form %s.text, lines=%d' % (reference_name, example_path, len(src)))
        my_examples[reference_name] = src
        if max_line != -1:
            for idx in my_examples.keys():
                my_examples[idx] = my_examples[idx][0:max_line]

        logger.info('[DATASET] Maximum %s context sequence length is set to %d' % (reference_name, max_context_len))
        logger.info('[DATASET] Maximum %s utterance sequence length is set to %d' % (reference_name, max_utterance_len))

        all_sub_tokens = defaultdict(list)
        for idx in range(len(my_examples[reference_name])):
            x = my_examples[reference_name][idx]
            context_items = x.split(self.context_split_notation)[0:max_context_len]
            token_set = set()
            for sidx in range(len(context_items)):
                y = context_items[sidx]
                items = y.split(self.utterance_split_notation)[0:max_utterance_len]
                token_set = token_set | set(items)
                context_items[sidx] = self.utterance_split_notation.join(items)
            all_sub_tokens[reference_name].append(token_set)
            my_examples[reference_name][idx] = self.context_split_notation.join(context_items)

        dmc_name = 'dmc_bdv_' + reference_name
        if dmc_name in field_dict:
            logger.info('[DATASET] Copy %s from %s, lines=%d' % (dmc_name, reference_name, len(src)))
            my_examples[dmc_name] = copy.deepcopy(my_examples[reference_name])

        # 创建适用于拷贝词表的部分
        logger.info('[DATASET] create extend vocabs for %s' % reference_name)
        if 'hierarchical_attentive_memory' in reference_params \
                and reference_params.hierarchical_attentive_memory.context.copy_mode is not False \
                and reference_params.hierarchical_attentive_memory.utterance.copy_mode is not False:
            local_vocabs = []
            generation_mode_fusion_mode = self.generation_mode_fusion_mode
            if generation_mode_fusion_mode == 'dynamic_mode_fusion':
                raise NotImplementedError()
                # for src_example in my_examples[reference_name]:
                #     local_vocabs.append(1)
            elif generation_mode_fusion_mode == 'dynamic_vocab':
                known_words = set(field_dict['tgt'].vocab.stoi.keys()) | set(self.get_special_tokens())
                for src_example_set in all_sub_tokens[reference_name]:
                    local_vocab = src_example_set - known_words
                    local_vocabs.append(local_vocab)
            return my_examples, {reference_name: (reference_params.dataset.max_len + 2, local_vocabs)}
        else:
            if 'hierarchical_attentive_memory' in reference_params:
                if reference_params.hierarchical_attentive_memory.utterance.copy_mode is not False:
                    assert reference_params.hierarchical_attentive_memory.context.copy_mode is not False
                if reference_params.hierarchical_attentive_memory.context.copy_mode is not False:
                    assert reference_params.hierarchical_attentive_memory.utterance.copy_mode is not False

            return my_examples, {}

    def create_fields(self, params, additional_params, tokenizer='default'):

        reference_params = self.reference_params
        reference_name = reference_params['name']

        # dynamic_vocab
        build_dynamic_field = additional_params['generation_mode_fusion_mode'] == 'dynamic_vocab'

        # 开始加载数据集
        vocab_mode = reference_params.dataset['vocab_mode']
        src_vocab_path = reference_params.dataset['vocab_path']

        src_field = SubTextField(include_lengths=True, context_split_notation=self.context_split_notation,
                                 utterance_split_notation=self.utterance_split_notation, max_len=self.max_len,
                                 init_token='<csos>', eos_token='<ceos>', batch_first=self.batch_first)

        # 创建词表
        if vocab_mode == 'separate':
            logger.info('[VOCAB] Loading %s-TEXT vocab from: %s' % (reference_name, src_vocab_path))
            src_vocab = load_vocab(src_vocab_path, self.get_special_tokens())
            src_field.vocab = src_vocab
        else:
            logger.info('[VOCAB] Do not load  %s-TEXT vocab' % reference_name)
            field_dict = additional_params['field_dict']
            src_field.vocab = field_dict['src'].vocab

        result = [
            (reference_name, src_field),
        ]

        if build_dynamic_field:
            dmc_src_field = SubTextField(include_lengths=False, context_split_notation=self.context_split_notation,
                                         utterance_split_notation=self.utterance_split_notation, max_len=self.max_len,
                                         init_token='<csos>', eos_token='<ceos>', dtype=torch.int32,
                                         batch_first=self.batch_first, flat_to_dmc=True, dynamic_vocab=True)
            field_dict = additional_params['field_dict']
            dmc_src_field.vocab = field_dict['tgt'].vocab
            result.append(('dmc_bdv_' + reference_name, dmc_src_field))



        return result
