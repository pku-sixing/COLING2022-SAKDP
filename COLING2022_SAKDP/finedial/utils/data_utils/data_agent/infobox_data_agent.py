import copy
from torchtext.data import Field

from finedial.utils.data_utils.input_helper import load_vocab
from finedial.utils.logging.logger_helper import logger
from finedial.ops.data_unit.Fields.BatchDynamicVocabField import BDVField

sp_tokens = ['<unk>', '<pad>', '<smr_key>', '<smr_value>', '<none>']
BOX_SUFFIX = 'box'


class InfoboxReferenceHelper:

    def __init__(self, params, reference_params):
        self.reference_params = reference_params
        self.suffix = reference_params.dataset.get('suffix', BOX_SUFFIX)
        self.batch_first = params.get('mp_gpu', 0) > 1
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
        max_len = reference_params.dataset['max_len']
        reference_name = reference_params['name']
        assert max_len > 0
        my_examples = dict()

        # Box
        my_examples[reference_name + '_key'] = []
        my_examples[reference_name + '_word'] = []
        my_examples[reference_name + '_fw_pos'] = []
        my_examples[reference_name + '_bw_pos'] = []
        my_examples[reference_name + '_tag'] = []
        box_path = example_path + '.' + self.suffix
        logger.info('[DATASET] Max instance  of %s  is limited to %d' % (reference_name, max_line))
        logger.info('[DATASET] Max sequence length  of %s  is limited to %d' % (reference_name, max_len))
        with open(box_path, 'r', encoding='utf-8') as fin:
            boxes = fin.readlines()
            if max_line > -1:
                boxes = boxes[0:max_line]
            for box in boxes:
                fields = box.strip('\r\n').split()[0:max_len]
                field_words = []
                field_keys = []
                forward_poses = []
                backward_poses = []
                pos_tags = []
                for field in fields:
                    items = field.split(':')
                    field_word = items[1]
                    sub_items = items[0].split('_')
                    field_key = sub_items[0]
                    forward_pos = int(sub_items[1])
                    backward_pos = int(sub_items[2])
                    pos_tag = sub_items[3]

                    field_words.append(field_word)
                    field_keys.append(field_key)
                    forward_poses.append(forward_pos)
                    backward_poses.append(backward_pos)
                    pos_tags.append(pos_tag)

                my_examples[reference_name + '_key'].append(field_words)
                my_examples[reference_name + '_word'].append(field_keys)
                my_examples[reference_name + '_fw_pos'].append(forward_poses)
                my_examples[reference_name + '_bw_pos'].append(backward_poses)
                my_examples[reference_name + '_tag'].append(pos_tags)

        my_examples[reference_name + '_key_word'] = copy.deepcopy(my_examples[reference_name + '_key'])
        my_examples[reference_name + '_word_word'] = copy.deepcopy(my_examples[reference_name + '_word'])

        dmc_name = 'dmc_bdv_' + reference_name + '_key'
        if dmc_name in field_dict:
            my_examples['dmc_bdv_' + reference_name + '_key'] = copy.deepcopy(my_examples[reference_name + '_key'])
            my_examples['dmc_bdv_' + reference_name + '_word'] = copy.deepcopy(my_examples[reference_name + '_word'])

        # 创建适用于拷贝词表的部分
        logger.info('[DATASET] create extend vocabs for %s' % reference_name)
        if reference_params.attentive_memory.copy_mode is not False:
            generation_mode_fusion_mode = self.generation_mode_fusion_mode
            if generation_mode_fusion_mode == 'dynamic_mode_fusion':
                local_vocabs = []
                for src_example in my_examples[reference_name + '_key']:
                    local_vocabs.append(1)
                local_key_vocabs = local_vocabs
                local_word_vocabs = local_vocabs
            elif generation_mode_fusion_mode == 'dynamic_vocab':
                known_words = set(field_dict['tgt'].vocab.stoi.keys()) | set(self.get_special_tokens())
                local_key_vocabs = []
                for src_example in my_examples[reference_name + '_key']:
                    local_vocab = set(src_example) - known_words
                    local_key_vocabs.append(local_vocab)
                local_word_vocabs = []
                for src_example in my_examples[reference_name + '_word']:
                    local_vocab = set(src_example) - known_words
                    local_word_vocabs.append(local_vocab)
            return my_examples, {
                reference_name + '_key': (reference_params.dataset.max_len + 1, local_key_vocabs),
                reference_name + '_word': (reference_params.dataset.max_len + 1, local_word_vocabs)
            }
        else:
            return my_examples, {}



    def create_fields(self, params, additional_params, tokenizer='default'):

        reference_params = self.reference_params
        reference_name = reference_params['name']
        field_dict = additional_params['field_dict']

        # dynamic_vocab
        build_dynamic_field = additional_params['generation_mode_fusion_mode'] == 'dynamic_vocab'

        # 开始加载数据集：Separate or Hybrid
        vocab_mode = reference_params.dataset['vocab_mode']
        field_key_vocab_path = reference_params.dataset['field_key_vocab_path']
        field_word_vocab_path = reference_params.dataset['field_word_vocab_path']
        field_tag_vocab_path = reference_params.dataset['field_tag_vocab_path']

        field_key_field = Field(tokenize=None, include_lengths=True, init_token='<smr_key>',
                                eos_token=None, batch_first=self.batch_first)
        field_word_field = Field(tokenize=None, include_lengths=False,
                                 init_token='<smr_value>', eos_token=None, batch_first=self.batch_first)
        field_tag_field = Field(tokenize=None, include_lengths=False,
                                init_token='<smr_tag>', eos_token=None, batch_first=self.batch_first)
        field_fw_pos_field = Field(tokenize=None, include_lengths=False, use_vocab=False,
                                   init_token=0, eos_token=None, unk_token=0, pad_token=0, batch_first=self.batch_first)
        field_bw_pos_field = Field(tokenize=None, include_lengths=False, use_vocab=False,
                                   init_token=0, eos_token=None, unk_token=0, pad_token=0, batch_first=self.batch_first)

        # 创建词表
        logger.info('[VOCAB] Loading %s-Infobox Field vocab from: Key=%s, Value=%s' %
                    (reference_name, field_key_vocab_path, field_word_vocab_path))
        field_vocab = load_vocab([field_key_vocab_path, field_word_vocab_path], self.get_special_tokens(),
                                 enable_duplication=True)
        field_key_field.vocab = field_vocab
        field_word_field.vocab = field_vocab
        logger.info('[VOCAB] Loading %s-Infobox Tag vocab from: Tag=%s' %
                    (reference_name, field_tag_vocab_path))
        tag_vocab = load_vocab(field_tag_vocab_path, self.get_special_tokens(),
                               enable_duplication=True)
        field_tag_field.vocab = tag_vocab

        result = [
            (reference_name + '_key', field_key_field),
            (reference_name + '_word', field_word_field),
            (reference_name + '_tag', field_tag_field),
            (reference_name + '_fw_pos', field_fw_pos_field),
            (reference_name + '_bw_pos', field_bw_pos_field),
        ]

        if vocab_mode != 'separate':
            # 需要专门加载Fields来保留Token ID，仅针对response和post
            field_key_word_field = Field(tokenize=None, include_lengths=True, init_token='<smr_key>',
                                         eos_token=None, batch_first=self.batch_first)
            field_word_word_field = Field(tokenize=None, include_lengths=False,
                                          init_token='<smr_value>', eos_token=None, batch_first=self.batch_first)
            field_key_word_field.vocab = field_dict['src'].vocab
            field_word_word_field.vocab = field_dict['src'].vocab
            result.append((reference_name + '_key_word', field_key_word_field))
            result.append((reference_name + '_word_word', field_word_word_field))

        if build_dynamic_field:
            dmc_key = BDVField(tokenize=None, include_lengths=False, dynamic_vocab=True,
                            init_token='<smr_key>', eos_token=None, batch_first=self.batch_first)
            dmc_word = BDVField(tokenize=None, include_lengths=False, dynamic_vocab=True,
                             init_token='<smr_value>', eos_token=None, batch_first=self.batch_first)

            dmc_key.vocab = field_dict['tgt'].vocab
            dmc_word.vocab = field_dict['tgt'].vocab

            result.append(('dmc_bdv_' + reference_name + '_key', dmc_word))
            result.append(('dmc_bdv_' + reference_name + '_word', dmc_key))

        return result
