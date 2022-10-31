import random
from collections import defaultdict

import torch
import torchtext
from torchtext.data import Dataset, BucketIterator, Iterator

from finedial.ops.data_unit.Fields.SubTextField import SubTextField
from finedial.utils.data_utils.data_agent.dialogue_data_agent import DialogueHelper
from finedial.utils.data_utils.data_agent.infobox_data_agent import InfoboxReferenceHelper
from finedial.utils.data_utils.data_agent.svo_data_agent import SVOReferenceHelper
from finedial.utils.data_utils.data_agent.text_data_agent import TextReferenceHelper
from finedial.utils.data_utils.data_agent.sub_text_data_agent import SubTextReferenceHelper
from finedial.utils.logging.logger_helper import logger


def transfer_batch_to_device(batch, device):
    for field in batch.fields:
        batch_field = getattr(batch, field)
        if torch.is_tensor(batch_field):
            field_device = batch_field.to(device)
            setattr(batch, field, field_device)
        elif isinstance(batch_field, tuple):
            new_tuple = [batch_field[0], batch_field[1]]
            new_tuple[0] = new_tuple[0].to(device)
            new_tuple[1] = new_tuple[1].to(device)
            new_tuple = tuple(new_tuple)
            setattr(batch, field, new_tuple)
    return batch


def _get_dataset(valid_input_helper, params, data_path, field_orders, field_dict):
    filters = None
    if 'filter' in params.dataset:
        filters = []
        filter_file = data_path + '.' + params.dataset.filter.filter_tags_suffix
        tags = open(filter_file, 'r+', encoding='utf-8').readlines()
        tags = [set(x.strip('\r\n').split()) for x in tags]
        max_line = params.dataset['max_line']
        if max_line != -1:
            tags = tags[0: max_line]
        valid_tags = set(params.dataset.filter.valid_tags)
        count = 0
        count_false = 0
        for tag in tags:
            count += 1
            filters.append(True)
            for valid_tag in valid_tags:
                if valid_tag not in tag:
                    filters[-1] = False
                    count_false += 1
                    break

        logger.info('[DATA Filter] %s, Total = %s, Valid = %s' % (data_path, count, count_false))


    my_examples = dict()
    valid_start_positions = dict()
    my_vocab_max_sizes = dict()
    for helper in valid_input_helper:
        local_examples, local_vocabs = helper.load_examples(data_path, field_dict)
        if filters is not None:
            for key, value in local_examples.items():
                assert len(value) == len(filters), '%s-%s' % (len(value), len(filters))
                new_value = []
                for f, v in zip(filters, value):
                    if f:
                        new_value.append(v)
                local_examples[key] = new_value
            for key, local_vocab in local_vocabs.items():
                t1, value = local_vocab
                assert len(value) == len(filters), '%s-%s' % (len(value), len(filters))
                new_value = []
                for f, v in zip(filters, value):
                    if f:
                        new_value.append(v)
                local_vocabs[key] = (t1, new_value)

        for sub_key in local_examples:
            assert sub_key not in my_examples, sub_key
            my_examples[sub_key] = local_examples[sub_key]
        for sub_key in local_vocabs:
            assert sub_key not in valid_start_positions, sub_key
            my_vocab_max_sizes[sub_key] = local_vocabs[sub_key][0]
            valid_start_positions[sub_key] = local_vocabs[sub_key][1]

    # 进行Copy对齐, 首先选择对齐策略
    if 'stepwise_decoder' in params.__dict__.keys():
        generation_mode_fusion_mode = params.stepwise_decoder['generation_mode_fusion_mode']
    else:
        generation_mode_fusion_mode = 'none'
    dmf_vocab_stop_list = []
    dmf_vocab_ptos_list = []
    if generation_mode_fusion_mode == 'dynamic_mode_fusion':
        copy_strategy = params.dataset.copy_strategy
        multi_select_strategy = params.dataset.multi_select_strategy
        dmf_vocab_size = 0
        dmf_vocab_offset = dict()
        assert len(params.dataset.copy_order) == len(my_vocab_max_sizes)
        for copy_source in params.dataset.copy_order:
            dmf_vocab_offset[copy_source] = dmf_vocab_size
            dmf_vocab_size += my_vocab_max_sizes[copy_source]

        # 合并Vocab PlaceholderToString , StringToPlaceholder
        for idx in range(len(my_examples['tgt'])):
            dmf_vocab_stop = defaultdict(list)
            dmf_vocab_ptos = dict()
            local_vocab = defaultdict(list)
            # Construct POS
            for copy_source in params.dataset.copy_order:
                local_vocab_offset = valid_start_positions[copy_source][idx] + dmf_vocab_offset[copy_source]
                if copy_source == 'query':
                    copy_source = 'src'
                for sub_pos, token in enumerate(my_examples[copy_source][idx].split()):
                    if copy_strategy == 'unk':
                        if token not in field_dict['tgt'].vocab.stoi:
                            local_vocab[token].append(sub_pos + local_vocab_offset)
                    else:
                        raise NotImplementedError()
            for token in local_vocab:
                for sub_pos in local_vocab[token]:
                    placeholder = '<dmc_%d>' % (sub_pos)
                    dmf_vocab_stop[token].append(placeholder)
                    dmf_vocab_ptos[placeholder] = token
            dmf_vocab_stop_list.append(dmf_vocab_stop)
            dmf_vocab_ptos_list.append(dmf_vocab_ptos)

        for idx in range(len(my_examples['tgt'])):
            tgt = my_examples['tgt'][idx].split()
            local_vocab = dmf_vocab_stop_list[idx]
            for sub_idx, token in enumerate(tgt):
                # 决定这个词是否拷贝
                if token not in local_vocab:
                    continue
                if copy_strategy == 'unk':
                    if token in field_dict['tgt'].vocab.stoi:
                        continue
                elif copy_strategy == 'valid':
                    pass
                else:
                    raise NotImplementedError()
                # 如何选择拷贝的词
                if multi_select_strategy == 'sample':
                    placeholder = random.sample(local_vocab[token], 1)[0]
                else:
                    raise NotImplementedError()
                tgt[sub_idx] = placeholder
            my_examples['tgt'][idx] = ' '.join(tgt)

    elif generation_mode_fusion_mode == 'dynamic_vocab':
        # 合并Vocab PlaceholderToString , StringToPlaceholder
        tgt_vocab = set(field_dict['tgt'].vocab.stoi.keys())
        for  key in valid_start_positions.keys():
            print(key, len(valid_start_positions[key]))
        for idx in range(len(my_examples['tgt'])):
            dmf_vocab_stop = dict()
            dmf_vocab_ptos = dict()
            for copy_source in params.dataset.copy_order:
                local_vocab = valid_start_positions[copy_source][idx] - tgt_vocab
                for token in local_vocab:
                    if token not in dmf_vocab_stop:
                        placeholder = '<dmc_%d>' % (len(dmf_vocab_stop))
                        dmf_vocab_stop[token] = placeholder
                        dmf_vocab_ptos[placeholder] = token
            dmf_vocab_stop_list.append(dmf_vocab_stop)
            dmf_vocab_ptos_list.append(dmf_vocab_ptos)

        for field_name in my_examples.keys():
            if field_name == 'tgt' or field_name.startswith("dmc_"):
                logger.info('[DynamicVocab] Adopting Dynamic Vocab For %s' % field_name)
                # 如果BDV 有效
                bdv_field_name = field_name.replace("dmc_", "bdv")
                if bdv_field_name in my_examples.keys():
                    logger.info('[DynamicVocab] Adopting Batch Dynamic Vocab For %s' % field_name)

                for idx in range(len(my_examples[field_name])):
                    if isinstance(my_examples[field_name][idx], str):
                        if isinstance(field_dict[field_name], SubTextField):
                            tgt = my_examples[field_name][idx].split(field_dict[field_name].context_split_notation)
                            tgt = [x.split(field_dict[field_name].utterance_split_notation) for x in tgt]
                        else:
                            tgt = my_examples[field_name][idx].split()
                    else:
                        tgt = my_examples[field_name][idx]
                    local_vocab = dmf_vocab_stop_list[idx]
                    for sub_idx, token in enumerate(tgt):
                        if isinstance(token, str):
                            #  直接就可以拷贝了
                            if token not in local_vocab:
                                continue
                            tgt[sub_idx] = local_vocab[token]
                        elif isinstance(token, list):
                            for cid, char in enumerate(token):
                                if char not in local_vocab:
                                    continue
                                token[cid] = local_vocab[char]
                            tgt[sub_idx] = field_dict[field_name].utterance_split_notation.join(token)
                        else:
                            raise NotImplementedError()
                    if isinstance(field_dict[field_name], SubTextField):
                        my_examples[field_name][idx] = field_dict[field_name].context_split_notation.join(tgt)
                    else:
                        my_examples[field_name][idx] = ' '.join(tgt)
            else:
                continue


    elif generation_mode_fusion_mode == 'none':
        pass
    else:
        raise NotImplementedError()

    # 生成DataSet
    example_filters = dict()
    if 'references' in params.keys():
        for reference in params.references:
            knowledge_length_filter = reference.dataset.get('knowledge_length_filter', -1)
            if knowledge_length_filter > 0:
                logger.info('[KNOWLEDGE_FILTER] set %s to %s' % (reference.name, knowledge_length_filter))
            else:
                continue
            reference_key_field = reference.name
            reference_type = reference.knowledge_type
            if reference_type == 'svo':
                reference_key_field += '_head'
            elif reference_type == 'infobox':
                reference_key_field += '_key'
            elif reference_type == 'text':
                continue
            else:
                raise NotImplementedError()
            example_filters[reference_key_field] = knowledge_length_filter

    # 创造初始的输入
    lines = []
    raw_num = len(my_examples['tgt'])
    for idx in range(raw_num):
        line = []
        valid_flag = True
        for field in field_orders:
            field_name = field[0]
            item = my_examples[field_name][idx]
            if field_name in example_filters:
                if len(item) < example_filters[field_name]:
                    valid_flag = False
                    break

            line += [item]
        if valid_flag:
            lines.append(line)
            assert len(line) == len(field_orders), '%s-%s' % (len(line), len(field_orders))

    examples = []
    for line in lines:
        example = torchtext.data.Example.fromlist(data=line, fields=field_orders)
        examples.append(example)
    dataset = Dataset(examples=examples, fields=field_orders)

    return dataset, dmf_vocab_ptos_list


def load_dataset(params, is_eval=False, dmf_vocab=False, batch_size=None, no_shuffle=False):
    logger.info("[DATASET] Preparing dataset...")
    if batch_size is None:
        batch_size = params.dataset['batch_size']
    logger.info('[DATASET] Batch size is set to %d' % batch_size)
    if 'infer' in params and 'batch_size' in params.infer:
        infer_batch_size = params.infer.batch_size
    else:
        infer_batch_size = batch_size
    logger.info('[DATASET] Infer Batch size is set to %d' % infer_batch_size)

    # 是否构造动态词表
    if 'stepwise_decoder' in params.__dict__.keys():
        generation_mode_fusion_mode = params.stepwise_decoder['generation_mode_fusion_mode']
    else:
        generation_mode_fusion_mode = 'none'
    assert generation_mode_fusion_mode in ['dynamic_mode_fusion', 'dynamic_vocab', 'none']

    # 所有输入的控制器
    dialog_helper = DialogueHelper(params)
    valid_input_helper = [dialog_helper]
    if 'references' in params.keys():
        for reference in params.references:
            if reference.knowledge_type == 'text':
                valid_input_helper.append(TextReferenceHelper(params, reference))
            elif reference.knowledge_type == 'sub_text':
                valid_input_helper.append(SubTextReferenceHelper(params, reference))
            elif reference.knowledge_type == 'svo':
                valid_input_helper.append(SVOReferenceHelper(params, reference))
            elif reference.knowledge_type == 'infobox':
                valid_input_helper.append(InfoboxReferenceHelper(params, reference))
            else:
                raise NotImplementedError()
    if 'auxiliary_inputs' in params.keys():
        for auxiliary_input in params.auxiliary_inputs:
            if auxiliary_input.knowledge_type == 'text':
                valid_input_helper.append(TextReferenceHelper(params, auxiliary_input))

    # 获得拷贝词表数量
    max_copy_vocab_size = 0
    for helper in valid_input_helper:
        max_copy_vocab_size += helper.get_local_vocab_size(params)

    # 创建Fields
    fields = []
    field_dict = {}

    # Additional params
    additional_params = dict()

    additional_params['max_copy_vocab_size'] = max_copy_vocab_size
    additional_params['generation_mode_fusion_mode'] = generation_mode_fusion_mode
    additional_params['field_dict'] = field_dict

    for helper in valid_input_helper:
        sub_fields = helper.create_fields(params, additional_params)
        fields += sub_fields
        for x in sub_fields:
            field_dict[x[0]] = x[1]

    # Field Orders & Update Fields:
    field_orders = []
    for field in fields:
        if field_dict[field[0]] is None:
            logger.info('[FIELD] Removing the invalid field %s:' % field[0])
            del field_dict[field[0]]
            assert field[0] not in field_dict
        else:
            logger.info('[FIELD] Keep the valid field %s:' % field[0])
            field_orders.append(field)

    # 加载数据
    val_data_path_prefix = params.dataset['val_data_path_prefix']
    val_set, val_dmf_vocab = _get_dataset(valid_input_helper, params, val_data_path_prefix,
                                          field_orders=field_orders, field_dict=field_dict)
    test_data_path_prefix = params.dataset['test_data_path_prefix']
    test_set, test_dmf_vocab = _get_dataset(valid_input_helper, params, test_data_path_prefix,
                                            field_orders=field_orders, field_dict=field_dict)

    device = 'cuda' if params['cuda'] else 'cpu'
    if 'src' in field_dict:
        sort_key = lambda x: len(x.tgt) + len(x.src) * (params.dataset['max_tgt_len'] + 5)
    elif 'context' in field_dict:
        def sort(x):
            lens = []
            lens.append(len(x.tgt))
            if 'concept_net_head' in field_dict:
                lens.append(len(x.concept_net_head))
            if 'wiki_text' in field_dict:
                lens.append(len(x.wiki_text))
            if 'wiki_box_key' in field_dict:
                lens.append(len(x.wiki_box_key))
            if 'context' in field_dict:
                lens.append(len(x.context))
            max_len = 205
            res = 0
            for len_i in lens:
                res = res * max_len + len_i
            return res
        sort_key = lambda x: sort(x)

    else:
        sort_key = lambda x: len(x.tgt)

    if dmf_vocab:
        field_dict['dmf_vocab'] = {
            'val': val_dmf_vocab,
            'test': test_dmf_vocab
        }
    else:
        del val_dmf_vocab, test_dmf_vocab

    dialogue_sp_token_dicts = dialog_helper.get_special_token_dict()

    if not is_eval:
        logger.info('[DATASET] Training Mode')
        training_data_path_prefix = params.dataset['training_data_path_prefix']
        training_set, train_dmf_vocab = _get_dataset(valid_input_helper, params, training_data_path_prefix,
                                                     field_orders=field_orders, field_dict=field_dict)

        if dmf_vocab:
            field_dict['dmf_vocab']['train'] = train_dmf_vocab
        else:
            del train_dmf_vocab

        if no_shuffle:
            train_iter = BucketIterator(training_set, batch_size=batch_size, repeat=False, shuffle=False,
                                        sort_key=sort_key, sort=False, train=False, sort_within_batch=True,
                                        device=device)
        else:
            train_iter = BucketIterator(training_set, batch_size=batch_size, repeat=False, shuffle=True,
                                        sort_key=sort_key, sort=False, train=True, sort_within_batch=True,
                                        device=device)
        val_iter = Iterator(val_set, batch_size=infer_batch_size, repeat=False, shuffle=False,
                            sort_key=sort_key, sort=False, train=False, sort_within_batch=True,
                            device=device)
        test_iter = Iterator(test_set, batch_size=infer_batch_size, repeat=False, shuffle=False,
                             sort_key=sort_key, sort=False, train=False, sort_within_batch=False,
                             device=device)
        logger.info("[TRAIN]: #Batches=%d (#Cases:%d))" % (len(train_iter), len(train_iter.dataset)))
        logger.info("[VALIDATION]: #Batches=%d (#Cases:%d))" % (len(val_iter), len(val_iter.dataset)))
        logger.info("[TEST]: #Batches=%d (#Cases:%d))" % (len(test_iter), len(test_iter.dataset)))

        return train_iter, val_iter, test_iter, field_dict, dialogue_sp_token_dicts
    else:
        logger.info('[DATASET] Eval/Inference Mode')
        val_iter = Iterator(val_set, batch_size=infer_batch_size, repeat=False, shuffle=False,
                            sort_key=sort_key, sort=False, train=False, sort_within_batch=False,
                            device=device)
        test_iter = Iterator(test_set, batch_size=infer_batch_size, repeat=False, shuffle=False,
                             sort_key=sort_key, sort=False, train=False, sort_within_batch=False,
                             device=device)

        logger.info("[VALIDATION]: #Batches=%d (#Cases:%d))" % (len(val_iter), len(val_iter.dataset)))
        logger.info("[TEST]: #Batches=%d (#Cases:%d))" % (len(test_iter), len(test_iter.dataset)))
        return None, val_iter, test_iter, field_dict, dialogue_sp_token_dicts
