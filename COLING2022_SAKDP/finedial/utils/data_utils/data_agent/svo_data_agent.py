import copy
from collections import  OrderedDict
from torchtext.data import Field
from finedial.ops.data_unit.Fields.BatchDynamicVocabField import BDVField
from finedial.utils.data_utils.input_helper import load_vocab
from finedial.utils.logging.logger_helper import logger

sp_tokens = ['<unk>', '<pad>', '<smr_csk_head>', '<smr_csk_tail>', '<smr_csk_rel>', '#NH', '#NT']
CSK_SUFFIX = 'naf_fact'


class SVOReferenceHelper:

    def __init__(self, params, reference_params):
        self.reference_params = reference_params
        self.suffix = reference_params.dataset.get('suffix', CSK_SUFFIX)
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
        # 是否拷贝末尾端
        side_aware = reference_params.get('side_aware', True)
        assert max_len > 0
        my_examples = dict()

        # fact dict
        fact_dict_path = self.reference_params.dataset.fact_dict_path

        with open(fact_dict_path, 'r', encoding='utf-8') as fin:
            raw_facts = [' '.join(x.strip('\r\n').split()[-3:]) for x in fin.readlines()]
            logger.info('[DATASET] RAW SVO-%s Fact Base has %d facts ' % (reference_name, len(raw_facts)))
            fact_duplication_dict = OrderedDict()
            for fact_id, fact in enumerate(raw_facts):
                if fact not in fact_duplication_dict:
                    if len(fact.split()) != 3:
                        logger.info('[SVO] Ignore the fact [%s]' % fact)
                        continue
                    fact_duplication_dict[fact] = set()
                fact_duplication_dict[fact].add(fact_id)
            internal_fact_lists = fact_duplication_dict.keys()
            external_fact_id_to_internal_fact_id = dict()
            for internal_fact_id, key in enumerate(internal_fact_lists):
                external_fact_ids = fact_duplication_dict[key]
                for eid in external_fact_ids:
                    external_fact_id_to_internal_fact_id[eid] = internal_fact_id
            internal_fact_lists = [x.split() for x in internal_fact_lists]
            internal_head_entities = [x[0] for x in internal_fact_lists]
            internal_rel_entities = [x[1] for x in internal_fact_lists]
            internal_tail_entities = [x[2] for x in internal_fact_lists]
            logger.info(
                '[DATASET] DISTINCT SVO-%s Fact Base has %d facts ' % (reference_name, len(internal_head_entities)))

        # COMMONSENSE_ID
        head_entities = []
        rel_entities = []
        tail_entities = []
        logger.info('[DATASET] Max instance  of %s  is limited to %d' % (reference_name, max_line))
        logger.info('[DATASET] Max sequence length  of %s  is limited to %d' % (reference_name, max_len))
        with open(example_path + '.' + self.suffix, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            if max_line != -1:
                lines = lines[0:max_line]
            external_ids = [[int(y) for y in x.strip('\n').split()] for x in lines]
            logger.info(
                '[DATASET] Loaded reference %s form %s.csk, lines=%d' % (
                reference_name, example_path, len(external_ids)))
            logger.info('[DATASET] Maximum %s sequence length is set to %d' % (reference_name, max_len))
            for idx, external_id in enumerate(external_ids):
                internal_set = set()
                internal_id = []
                for eid in external_id:
                    iid = external_fact_id_to_internal_fact_id[eid]
                    if iid not in internal_set:
                        internal_set.add(iid)
                        internal_id.append(iid)
                    if len(internal_id) > max_len:
                        break
                head_entity = [internal_head_entities[iid] for iid in internal_id]
                rel_entity = [internal_rel_entities[iid] for iid in internal_id]
                tail_entity = [internal_tail_entities[iid] for iid in internal_id]
                head_entities.append(head_entity)
                rel_entities.append(rel_entity)
                tail_entities.append(tail_entity)

        my_examples[reference_name + '_head'] = head_entities
        my_examples[reference_name + '_rel'] = rel_entities
        my_examples[reference_name + '_tail'] = tail_entities

        if reference_name + '_head_word' in field_dict:
            my_examples[reference_name + '_head_word'] = copy.deepcopy(my_examples[reference_name + '_head'])
            my_examples[reference_name + '_tail_word'] = copy.deepcopy(my_examples[reference_name + '_tail'])

        if 'dmc_bdv_' + reference_name + '_head' in field_dict:
            assert side_aware
            my_examples['dmc_bdv_' + reference_name + '_head'] = copy.deepcopy(my_examples[reference_name + '_head'])
        if 'dmc_bdv_' + reference_name + '_tail' in field_dict:
            my_examples['dmc_bdv_' + reference_name + '_tail'] = copy.deepcopy(my_examples[reference_name + '_tail'])

        # 创建适用于拷贝词表的部分
        logger.info('[DATASET] create extend vocabs for %s' % reference_name)
        if reference_params.attentive_memory.copy_mode is not False:
            generation_mode_fusion_mode = self.generation_mode_fusion_mode
            if generation_mode_fusion_mode == 'dynamic_mode_fusion':
                local_vocabs = []
                for src_example in my_examples[reference_name]:
                    local_vocabs.append(1)
                local_head_vocabs = local_vocabs
                local_tail_vocabs = local_vocabs
            elif generation_mode_fusion_mode == 'dynamic_vocab':
                known_words = set(field_dict['tgt'].vocab.stoi.keys()) | set(self.get_special_tokens())
                if side_aware:
                    local_head_vocabs = []
                    for src_example in my_examples[reference_name + '_head']:
                        local_vocab = set(src_example) - known_words
                        local_head_vocabs.append(local_vocab)
                local_tail_vocabs = []
                for src_example in my_examples[reference_name + '_tail']:
                    local_vocab = set(src_example) - known_words
                    local_tail_vocabs.append(local_vocab)
            if side_aware:
                return my_examples, {
                    reference_name + '_head': (reference_params.dataset.max_len + 1, local_head_vocabs),
                    reference_name + '_tail': (reference_params.dataset.max_len + 1, local_tail_vocabs)
                }
            else:
                return my_examples, {
                    reference_name + '_tail': (reference_params.dataset.max_len + 1, local_tail_vocabs)
                }
        else:
            return my_examples, {}

    def create_fields(self, params, additional_params, tokenizer='default'):

        reference_params = self.reference_params
        reference_name = reference_params['name']
        field_dict = additional_params['field_dict']

        # dynamic_vocab
        build_dynamic_field = additional_params['generation_mode_fusion_mode'] == 'dynamic_vocab'
        build_dynamic_field = build_dynamic_field & reference_params.attentive_memory.copy_mode is not False

        # 是否进行Side-Aware的拷贝, 否则默认则是只拷贝tail的entity
        side_aware = reference_params.get('side_aware', True)

        # 开始加载数据集：Separate or Hybrid
        vocab_mode = reference_params.dataset['vocab_mode']
        entity_vocab_path = reference_params.dataset['entity_vocab_path']
        relation_vocab_path = reference_params.dataset['relation_vocab_path']

        csk_head_entity_field = Field(tokenize=None, include_lengths=True, init_token='<smr_csk_head>',
                                      eos_token=None, batch_first=self.batch_first)
        csk_tail_entity_field = Field(tokenize=None, include_lengths=False,
                                      init_token='<smr_csk_tail>', eos_token=None, batch_first=self.batch_first)
        csk_relation_entity_field = Field(tokenize=None, include_lengths=False,
                                          init_token='<smr_csk_rel>', eos_token=None, batch_first=self.batch_first)

        # 创建词表
        logger.info('[VOCAB] Loading %s-CSK vocab from: Entity=%s, Relation=%s' %
                    (reference_name, entity_vocab_path, relation_vocab_path))
        entity_relation_vocab = load_vocab([entity_vocab_path, relation_vocab_path], self.get_special_tokens())
        csk_relation_entity_field.vocab = entity_relation_vocab
        csk_tail_entity_field.vocab = entity_relation_vocab
        csk_head_entity_field.vocab = entity_relation_vocab

        result = [
            (reference_name + '_head', csk_head_entity_field),
            (reference_name + '_rel', csk_relation_entity_field),
            (reference_name + '_tail', csk_tail_entity_field),
        ]

        if vocab_mode != 'separate':
            # 需要专门加载Fields来保留Token ID，仅针对response和post
            csk_head_word_field = Field(tokenize=None, include_lengths=False,
                                        init_token='<smr_csk_head>', eos_token=None, batch_first=self.batch_first)
            csk_tail_word_field = Field(tokenize=None, include_lengths=False,
                                        init_token='<smr_csk_tail>', eos_token=None, batch_first=self.batch_first)
            csk_head_word_field.vocab = field_dict['src'].vocab
            csk_tail_word_field.vocab = field_dict['src'].vocab
            result.append((reference_name + '_head_word', csk_head_word_field))
            result.append((reference_name + '_tail_word', csk_tail_word_field))

        if build_dynamic_field:
            if side_aware:
                dmc_head = BDVField(tokenize=None, include_lengths=False, dynamic_vocab=True,
                                 init_token='<smr_csk_head>', eos_token=None, batch_first=self.batch_first)

                dmc_head.vocab = field_dict['tgt'].vocab
                result.append(('dmc_bdv_' + reference_name + '_head', dmc_head))

            dmc_tail = BDVField(tokenize=None, include_lengths=False,dynamic_vocab=True,
                             init_token='<smr_csk_head>', eos_token=None, batch_first=self.batch_first)
            dmc_tail.vocab = field_dict['tgt'].vocab
            result.append(('dmc_bdv_' + reference_name + '_tail', dmc_tail))

        return result
