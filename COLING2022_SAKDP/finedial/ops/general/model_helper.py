import copy
import pickle
import pprint
import random
from collections import defaultdict

import numpy as np
import torch.nn as nn
from torch import optim
import torch
import os

from finedial.utils.data_utils.param_helper import ParamDict
from finedial.utils.logging.logger_helper import logger


def set_seed(seed=12345):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_optimizer_fn(optimizer='Adam'):
    if optimizer == 'Adam':
        return optim.Adam
    elif optimizer == 'AdamW':
        return optim.AdamW
    raise NotImplementedError


# 权重初始化，默认xavier
def adjust_learning_rate(optimizer, rate=0.5, min_value=None):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    new_lrs = []
    for param_group in optimizer.param_groups:
        new_lr = param_group['lr'] * rate
        if min_value is not None:
            new_lr = max(new_lr, min_value)
        logger.info('[LEARNING RATE] adjusting %f to %f ' % (param_group['lr'], new_lr))
        param_group['lr'] = new_lr
        new_lrs.append(new_lr)
    return new_lrs[0]


def name_a_generation(args, mode):
    test_file_name = args.test_data_path_prefix
    if test_file_name.find('/') > -1:
        mode = test_file_name.split('/')[-1]
    else:
        mode = test_file_name.split('\\')[-1]
    print('mode name => ', mode)

    if args.disable_unk_output:
        mask_unk = 'masked'
    else:
        mask_unk = 'std'
    try:
        if args.random_test_field_order:
            mask_unk += '_roder'
        if args.beam_length_penalize == 'avg':
            mask_unk += '_avg'
        if args.repeat_index > 0:
            mask_unk += '_%d' % args.repeat_index

    except:
        pass
    return '%s_B%d_D%.2f_%s' % (mode, args.beam_width, args.diverse_decoding, mask_unk)


def reset_learning_rate(optimizer, lr_rate):
    assert len(optimizer.param_groups) == 1
    for param_group in optimizer.param_groups:
        new_lr = lr_rate
        logger.info('[LEARNING RATE] adjusting %f to %f ' % (param_group['lr'], new_lr))
        param_group['lr'] = new_lr
        return param_group['lr']


def show_parameters(model):
    trainable_param_counter = defaultdict(float)
    logger.info('Trainable Parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            prefix = name.split('.')[0]
            trainable_param_counter[prefix] += param.nelement()
            logger.info('{}-{}-{}-{}'.format(name, param.shape, param.dtype, param.device))
    logger.info('-------------')
    trainable_sum = 0
    for key in trainable_param_counter.keys():
        logger.info('[PARAMS-COUNTING] #%s:%.2fM' % (key, trainable_param_counter[key] / 1e6))
        trainable_sum += trainable_param_counter[key]
    logger.info('[PARAMS-SUM] #%s:%.2fM' % ('Trainable', trainable_sum / 1e6))

    non_trainable_param_counter = defaultdict(float)
    logger.info('###########')
    logger.info('Non-Trainable Parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            prefix = name.split('.')[0]
            non_trainable_param_counter[prefix] += param.nelement()
            logger.info('{}-{}-{}-{}'.format(name, param.shape, param.dtype, param.device))
    logger.info('-------------')
    non_trainable_sum = 0
    for key in non_trainable_param_counter.keys():
        logger.info('[PARAMS-COUNTING] #%s:%.2fM' % (key, non_trainable_param_counter[key] / 1e6))
        non_trainable_sum += non_trainable_param_counter[key]
    logger.info('[PARAMS-SUM] #%s:%.2fM' % ('Non-Trainable', non_trainable_sum / 1e6))
    logger.info('-------------')
    logger.info('[PARAMS-SUM] #%s:%.2fM' % ('Total', (trainable_sum + non_trainable_sum) / 1e6))


def load_pretrain_trans_embeddings(embedding, filed, vocab_paths, embedding_paths, dim=100, suffix=None,
                                   ignore_sp_tokens_at_beginning=True):
    """
        embedding_paths/vocab_paths: 第一个Entity， 第二个Relation
    """
    assert len(embedding_paths) == 2 and len(vocab_paths) == 2
    vocabs = []
    pretrain_embeddings = []

    logger.info('[PRE-TRANS-EMBED] Try to merge entity embedding %s and relation %s with vocabs' % (embedding_paths[0],
                                                                                                    embedding_paths[1]))
    # 合并并生成成中间的
    for embedding_path, vocab_path in zip(embedding_paths, vocab_paths):
        logger.info('[PRE-TRANS-EMBED] Loading %s' % vocab_path)
        vocab = open(vocab_path, 'r+', encoding='utf-8').readlines()
        vocab = [x.strip('\r\n') for x in vocab]
        if ignore_sp_tokens_at_beginning:
            idx = 0
            while vocab[idx].startswith('#'):
                logger.info('Ignoring entity/relation : %s' % vocab[idx])
                idx += 1
            vocab = vocab[idx:]

        pretrain_embed = open(embedding_path, 'r+', encoding='utf-8').readlines()
        pretrain_embed = [x.strip('\r\n') for x in pretrain_embed]
        assert len(pretrain_embed) == len(vocab)

        pretrain_embeddings += pretrain_embed
        vocabs += vocab

    logger.info('[PRE-TRANS-EMBED] Entity_Rel embedding is saved to %s' % (embedding_paths[0] + '.uni'))
    with open(embedding_paths[0] + '.uni', 'w+', encoding='utf-8') as fout:
        for entity, embed in zip(vocabs, pretrain_embeddings):
            digits = embed.split()
            assert len(digits) == dim
            tmp = [entity] + digits
            fout.write('%s\n' % ' '.join(tmp))

    load_pretrain_embeddings(embedding, filed, embedding_paths[0] + '.uni', dim, char2word='sum', suffix=suffix)


def load_pretrain_embeddings(embedding, filed, embedding_path, dim=200, char2word='sum', suffix=None, debug=False):
    weights = embedding.weight.data.cpu().numpy()
    total_word = len(weights)
    flag = True
    loaded_vecs = None
    if suffix is not None:
        logger.info('[PRE-EMBED] Try to load embedding for %s from the cache %s.embed' % (str(filed), suffix))
        try:
            loaded_vecs = pickle.load(open('embedding_cache/%s.embed' % suffix, 'rb'))
            logger.info('[PRE-EMBED] Successfully loaded embedding for %s from the cache .%s' % (str(filed), suffix))
        except FileNotFoundError:
            loaded_vecs = None
            logger.info('[PRE-EMBED] Failed to load embedding for %s from the cache .%s' % (str(filed), suffix))

    if loaded_vecs is None:
        loaded_vecs = dict()
        logger.info('[PRE-EMBED] Loading for %s, Char2Word: %s' % (str(filed), str(char2word)))
        token_set = set()
        for word in filed.vocab.stoi:
            token_set.add(word)
            if char2word != 'none':
                for char in word:
                    token_set.add(char)

        with open(embedding_path, 'r+', encoding='utf-8') as fin:
            line = fin.readline()
            line_counter = 0
            while line is not None and len(line) > 0:
                if line_counter % 100000 == 0:
                    print("reading :%d line" % line_counter)
                line_counter += 1
                items = line.strip('\r\n').split()
                if len(items) != dim + 1:
                    print('invalid_dim ', len(items), dim + 1)
                    line = fin.readline()
                    continue
                word = items[0]
                word_weights = items[1:]
                if word in token_set:
                    word_weights = np.array([float(x) for x in word_weights])
                    loaded_vecs[word] = word_weights
                    flag = True
                    if len(loaded_vecs) % 1000 == 0 and flag:
                        logger.info('[PRE-EMBED] Loaded/TotalWord/TotalTokens:  %d/%d/%d' %
                                    (len(loaded_vecs), total_word, len(token_set)))
                        flag = False
                line = fin.readline()
        assert suffix is not None
        if os.path.exists("embedding_cache") is False:
            os.makedirs("embedding_cache")
        pickle.dump(loaded_vecs, open('embedding_cache/%s.embed' % suffix, 'wb'), )
    logger.info('[PRE-EMBED] Loaded Token/Total: %d/%d' % (len(loaded_vecs), total_word))

    pretrained_weight = np.zeros([len(embedding.weight.data), dim])
    if debug:
        logger.info(str(loaded_vecs.keys()))
        logger.info(str(filed.vocab.stoi.keys()))
    load_count = 0
    generate_count = 0
    for i in range(total_word):
        word = filed.vocab.itos[i]
        if word in loaded_vecs:
            load_count += 1
            pretrained_weight[i] = loaded_vecs[word]
        else:
            if char2word == 'none':
                pretrained_weight[i] = weights[i]
            elif char2word == 'avg' or char2word == 'sum':
                tmp = np.zeros([dim], dtype=np.float)
                tmp_flag = False
                for char in word:
                    if char in loaded_vecs:
                        tmp += loaded_vecs[char]
                        tmp_flag = True
                    else:
                        # 加一个随机的，这里已经初始化
                        tmp += weights[i]

                if tmp_flag:
                    generate_count += 1
                    if char2word == 'avg' and len(word) > 0:
                        tmp /= len(word)
                    pretrained_weight[i] = tmp
                else:
                    pretrained_weight[i] = weights[i]
            else:
                raise NotImplementedError()

    logger.info('[PRE-EMBED] Loaded/Generated/Word/Total: %d/%d/%d' % (load_count, generate_count, total_word))
    embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))


def try_restore_or_load_model(params, field_dict, config_path, model, optimizer, states, best_model):
    try:
        try_restore_model(params.model_path, params.experiment_name, model, optimizer, states, best_model, params.cuda)
    except:
        model_path = os.path.join(params.model_path, params.experiment_name)
        logger.info("[PARAM] Using fresh params")
        os.system('mkdir %s' % model_path)
        os.system('rm %s/source_code.zip' % model_path)
        os.system('zip -r %s/source_code.zip finedial/ %s' % (model_path, config_path))
        print('zip -r %s/source_code.zip finedial/ %s' % (model_path, config_path))

        init_network(model, params.training["init"])

        loaded_flags = set()
        logger.info('[Embedding] flag num from the given model %d' % (len(model.get_token_embeddings())))
        for item in model.get_token_embeddings():
            field_name = item[0]
            embedding = item[1]
            if field_name not in params.embedding['valid_init_flags']:
                logger.info('[Embedding] Ignore initializing embedding for field %s' % field_name)
                continue
            logger.info('[Embedding]  Initializing embedding for field %s' % field_name)
            load_pretrain_embeddings(embedding, field_dict[field_name], params.embedding['pre_embed_file']
                                     , dim=params.embedding['pre_embed_dim'],
                                     char2word=params.embedding['pre_embed_char2word'],
                                     suffix=params.dataset['dataset_version'] + '_' + field_name)
            loaded_flags.add(field_name)

        for item in model.get_trans_embeddings():
            field_name = item[0]
            embedding = item[1]
            vocab_paths = item[2]
            embedding_paths = item[3]
            dim = item[4]
            if field_name not in params.embedding['valid_init_flags']:
                logger.info('[Embedding] Ignore initializing embedding for field %s' % field_name)
                continue

            load_pretrain_trans_embeddings(embedding, field_dict[field_name], vocab_paths, embedding_paths, dim,
                                           suffix='trans_' + params.dataset['dataset_version'] + field_name)
            loaded_flags.add(field_name)
        logger.info('[Embedding] loaded pre-training flags: %s' % ' '.join(loaded_flags))
        logger.info('[Embedding] defined pre-training flags: %s' % ' '.join(params.embedding['valid_init_flags']))
        assert len(loaded_flags - set(params.embedding['valid_init_flags'])) == 0
        assert len(set(params.embedding['valid_init_flags']) - loaded_flags) == 0

        # 最后才加载预训练模块：
        if 'pretrained_modules' in params:
            for module in params['pretrained_modules']:
                logger.info('Loading pretrained module :%s' % module.name)
                try_to_load_pretrained_modules(module.name, module.module_dir, model,
                                               True, params.cuda)

        if params.cuda:
            logger.info('[PARAM] Enabling CUDA')
            assert torch.cuda.is_available()
            model = model.to('cuda')
            show_parameters(model)


def try_to_load_pretrained_modules(module_name, model_dir, model, best_model, cuda=False):
    logger.info("[MODEL] Loading the pretrain model from checkpoint...")
    model_path = os.path.join(model_dir)
    if best_model:
        model_path = os.path.join(model_path, 'best_model')
    else:
        model_path = os.path.join(model_path, 'check_point')
    if not os.path.isdir(model_path):
        logger.info('[CHECKPOINT] No checkpoint is found! Dir is not existed :%s' % model_path)
        raise FileNotFoundError()
    files = os.listdir(model_path)
    files = sorted(files, reverse=False)
    for file in files:
        if file[-3:] == '.pt':
            model_name = '%s/%s' % (model_path, file)
            checkpoint = torch.load(model_name)
            current_state_dict = model.state_dict()
            loaded_state_dict = checkpoint['model']
            pretrained_dict = {}
            discarded_dict = {}
            missed_dict = {}
            for k, v in loaded_state_dict.items():
                if k in current_state_dict and v.shape == current_state_dict[k].shape:
                    # logger.info('[PretrainedModules] Module=%s, load=%s' % (module_name, k))
                    pretrained_dict[k] = v
                elif k == 'tgt_token_embedding.weight' or k == 'src_token_embedding.weight':
                    current_v = current_state_dict[k].to(v.device)
                    pretrain_v = v
                    current_num, current_dim = current_v.size()
                    pretrain_num, pretrain_dim = pretrain_v.size()

                    assert current_num > pretrain_num and current_dim == pretrain_dim
                    new_v = torch.cat([pretrain_v, current_v[pretrain_num:]], dim=0)

                    pretrained_dict[k] = new_v
                else:
                    discarded_dict[k] = v
                    # logger.info('[PretrainedModules] Module=%s, discard=%s' % (module_name, k))

            for k, v in current_state_dict.items():
                if k not in pretrained_dict and k not in discarded_dict:
                    missed_dict[k] = v
                    # logger.info('[PretrainedModules] Module=%s, miss=%s' % (module_name, k))

            logger.info('[Pre-trained Modules] Module=%s, loaded=%s' % (module_name,
                                                                        pprint.pformat(pretrained_dict.keys())))
            logger.info('[Pre-trained Modules] Module=%s, discarded=%s' % (module_name,
                                                                           pprint.pformat(discarded_dict.keys())))
            logger.info('[Pre-trained Modules] Module=%s, missed=%s' % (module_name,
                                                                        pprint.pformat(missed_dict.keys())))

            current_state_dict.update(pretrained_dict)
            model.load_state_dict(current_state_dict)

            logger.info('[CHECKPOINT] Loaded module:[%s] params from  :%s' % (module_name, model_name))
            if cuda:
                logger.info('[PARAM] Enabling CUDA')
                assert torch.cuda.is_available()
                model = model.to('cuda')
                show_parameters(model)
            return
    logger.info('[CHECKPOINT] No checkpoint is found in :%s' % model_path)
    raise FileNotFoundError()


def try_restore_model(model_dir, experiment_name, model, optimizer, states, best_model, cuda=False):
    logger.info("[MODEL] Loading the model from checkpoint...")
    model_path = os.path.join(model_dir, experiment_name)
    if best_model:
        model_path = os.path.join(model_path, 'best_model')
    else:
        model_path = os.path.join(model_path, 'check_point')
    if not os.path.isdir(model_path):
        logger.info('[CHECKPOINT] No checkpoint is found! Dir is not existed :%s' % model_path)
        raise FileNotFoundError()
    files = os.listdir(model_path)
    files = sorted(files, reverse=False)
    for file in files:
        if file[-3:] == '.pt':
            model_name = '%s/%s' % (model_path, file)
            checkpoint = torch.load(model_name)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['model'])
            for key in checkpoint['states']:
                states[key] = checkpoint['states'][key]
            logger.info('[CHECKPOINT] Loaded params from  :%s' % model_name)
            if cuda:
                logger.info('[PARAM] Enabling CUDA')
                assert torch.cuda.is_available()
                model = model.to('cuda')
                show_parameters(model)

                if optimizer is not None:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()
            return
    logger.info('[CHECKPOINT] No checkpoint is found in :%s' % model_path)
    raise FileNotFoundError()


def save_model(model_path, experiment_name, epoch, val_loss, model, optimizer, arguments, states, best_model=True,
               clear_history=True, data_parallel=False):
    if best_model:
        model_path = os.path.join(model_path, experiment_name, 'best_model')
    else:
        model_path = os.path.join(model_path, experiment_name, 'check_point')
    if not os.path.isdir(model_path):
        logger.info('[CHECKPOINT] Creating model file:%s' % model_path)
        os.makedirs(model_path)
    model_name = '%s/model_%d_%d.pt' % (model_path, epoch, int(val_loss * 100))

    arguments_dict = {}
    if isinstance(arguments, dict) is False:
        arguments = vars(arguments)
        for key, value in arguments.items():
            arguments_dict[key] = value
    else:
        arguments_dict = arguments

    model_state = {
        'arguments': arguments_dict,
        'model': model.state_dict() if not data_parallel else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'states': states
    }
    torch.save(model_state, model_name)

    logger.info('[CHECKPOINT] Model has been saved to :%s' % model_name)
    if clear_history:
        logger.info('[CHECKPOINT] Removing old checkpoints')
        files = os.listdir(model_path)
        for file in files:
            file_name = '%s/%s' % (model_path, file)
            if file_name != model_name:
                logger.info('[CHECKPOINT] Removing %s' % file_name)
                try:
                    os.remove(file_name)
                except Exception as e:
                    print(e)


def init_network(model, method='xavier'):
    """
    :param model:
    :param method:
    :param seed:
    :return:
    """
    logger.info('[INIT] Initializing parameters: %s' % method)
    for name, w in model.named_parameters():
        if name.find('pretrained') > -1:
            logger.info('[INIT] SKIP to initialize %s ' % name)
            continue
        if 'bias' in name:
            nn.init.constant_(w, 0)
        else:
            if method == 'uniform' or w.dim() == 1:
                nn.init.uniform_(w, -0.1, 0.1)
            elif method == 'xavier':
                nn.init.xavier_uniform_(w)
            elif method == 'xavier_normal':
                nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_uniform_(w)
            elif method == 'kaiming_normal':
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)


def sequence_mask_fn(lengths, maxlen=None, dtype=torch.bool, mask_first=False):
    """

    :param lengths: [seq_len]
    :param maxlen:
    :param dtype:
    :return:
    """
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1, device=lengths.device, requires_grad=False)
    matrix = torch.unsqueeze(lengths, dim=-1)
    # mask = row_vector < matrix
    mask = row_vector.lt(matrix)
    if mask_first:
        mask[:, 0:1] = False
    mask = mask.type(dtype)
    return mask


def select_time_first_sequence_embedding(inputs, index):
    """

    :param inputs: (src_len, batch_size, dim)
    :param dim0's index: (batch_size)
    :return:
    """
    src_len, batch_size, dim = inputs.shape
    inputs = inputs.view(src_len * batch_size, dim)
    index_with_offset = index * batch_size + torch.arange(0, batch_size, dtype=torch.long, device=index.device)
    outputs = inputs.index_select(dim=0, index=index_with_offset)
    return outputs


def compute_bow_loss(bow_prob_dists, tgt_seq, valid_bow_id_start=1):
    """

    :param bow_prob_dists: (batch_size, vocab_size)
    :param tgt_seq: (batch_size, seq_len)
    :return:
    """
    batch_size, vocab_size = bow_prob_dists.size()
    flatten_inputs = bow_prob_dists.view(-1)
    index_offset = torch.arange(0, batch_size, device=bow_prob_dists.device).unsqueeze(1) * vocab_size
    # 防止dmc富豪
    index = torch.where(tgt_seq < vocab_size, tgt_seq, torch.zeros_like(tgt_seq))
    index_offset = index + index_offset
    flatten_index = index_offset.reshape(-1)
    # => (batch_size, seq_len)
    res = flatten_inputs.index_select(dim=0, index=flatten_index).view(batch_size, -1)
    res = torch.where(index >= valid_bow_id_start, torch.log(res + 1e-10), torch.zeros_like(res))
    valid_len = torch.where(index >= valid_bow_id_start, torch.ones_like(res), torch.zeros_like(res))
    valid_len = valid_len.sum(dim=1)
    loss = res.sum(dim=1) / valid_len
    return -loss.mean()


def create_beam(tensor, beam_width, batch_dim):
    """

    :param tensor:
    :param beam_width:
    :param batch_dim:
    :return:
    """
    if isinstance(tensor, tuple):
        return tuple([create_beam(x, beam_width, batch_dim) for x in tensor])
    elif isinstance(tensor, list):
        return [create_beam(x, beam_width, batch_dim) for x in tensor]
    elif isinstance(tensor, dict):
        for key, value in tensor.items():
            tensor[key] = create_beam(value, beam_width, batch_dim)
        return tensor
    else:
        if tensor is None:
            return tensor
        return torch.repeat_interleave(tensor, beam_width, batch_dim)


def generate_copy_offset_matrix(target_vocab_size, dynamic_vocab, stop_word_num=100, fast_mode=False):
    """

    :param target_vocab_size: 目标词的大小
    :param dynamic_vocab: [Seq_len, batch]
    :param dynamic_vocab_len: [Seq_len]
    :return:
    """
    # Dynamic 每个Batch一样
    dynamic_vocab_size = dynamic_vocab.shape[0]
    batch_size = dynamic_vocab.shape[1]
    # [vocab_size]
    copy_matrix = torch.arange(0, target_vocab_size, device=dynamic_vocab.device, dtype=torch.int32)
    # [1,1,vocab_size]
    copy_matrix = copy_matrix.view(1, 1, -1)
    # [batch, dynamic_vocab, vocab_size]
    copy_matrix = copy_matrix.repeat([batch_size, dynamic_vocab_size, 1])
    # 不拷贝超过100的
    valid_dynamic_vocab = torch.where(dynamic_vocab > stop_word_num, dynamic_vocab,
                                      torch.zeros_like(dynamic_vocab))
    #  [batch, Seq_len, 1]
    valid_dynamic_vocab = valid_dynamic_vocab.transpose(0, 1).unsqueeze(-1).to(torch.int32)
    copy_matrix = copy_matrix == valid_dynamic_vocab
    # TODO 在这里直接进行了叠加,需要尝试一下是取最大或者其他什么吗？
    # [batch, seq_len, vocab_size]
    copy_matrix = copy_matrix.to(torch.float32)
    return copy_matrix


def generate_copy_offset_probs(batch_size, prob_output, raw_vocab_size, std_vocab_size, dynamic_vocab,
                               copy_matrix_cache=None):
    """
    需要给出最后完整的Prob_out的概率，以及dynamic vocab
    TODO 之后的优化版本，可以直接使用分离的版本进行测试
    """
    # if True:
    if dynamic_vocab.device.type.find('cpu') > -1:
        float_precision = torch.float32
    else:
        float_precision = torch.float32
    if copy_matrix_cache is None or copy_matrix_cache['batch_size'] != batch_size:
        # Dynamic 每个Batch一样
        copy_matrix = torch.arange(0, raw_vocab_size, device=dynamic_vocab.device, dtype=torch.int32)
        copy_matrix = copy_matrix.view(1, 1, -1)
        # [batch, dynamic_vocab, std_vocab] 每个Dynamic_Vocab所对应的真实词表的映射关系
        copy_matrix = copy_matrix.repeat([batch_size, std_vocab_size - raw_vocab_size, 1])
        valid_dynamic_vocab = torch.where(dynamic_vocab > 100, dynamic_vocab,
                                          torch.zeros_like(dynamic_vocab))
        valid_dynamic_vocab = valid_dynamic_vocab.unsqueeze(-1).to(torch.int32)
        copy_matrix = copy_matrix == valid_dynamic_vocab
        # TODO 在这里直接进行了叠加,需要尝试一下是取最大或者其他什么吗？
        copy_matrix = copy_matrix.to(float_precision)
        copy_matrix_cache = {
            'batch_size': batch_size,
            'copy_matrix': copy_matrix,
        }
    else:
        copy_matrix = copy_matrix_cache['copy_matrix']

    copied_probs = prob_output[:, raw_vocab_size:].unsqueeze(1)
    copied_probs = copied_probs.to(float_precision)
    # [batch, 1, dynamic_vocab] * [batch, dynamic_vocab, std_vocab] => [batch, 1, std_vocab]
    probs_with_offset = torch.bmm(copied_probs, copy_matrix)
    probs_with_offset = probs_with_offset.squeeze(1)
    probs_with_offset[:, 0:100] = 0.0
    prob_output_with_copy = torch.cat(
        [prob_output[:, 0: raw_vocab_size] + probs_with_offset,
         prob_output[:, raw_vocab_size:]], -1)
    return prob_output_with_copy, copy_matrix_cache


def select_dynamic_embeddings_from_time_first_idx(char_embeddings, char_idx):
    """

    :param char_embeddings: [dynamic_vocab_size, batch_size, dim]
    :param char_idx: [seq_len, batch_size]
    :return: [seq_len, batch_size, dim]
    """
    # => [batch_size, vocab_size, dim]
    char_embeddings = char_embeddings.permute(1, 0, 2)
    batch_size, vocab_size, dim = char_embeddings.shape
    # idx [seq_len, batch_size]
    char_idx = char_idx
    # => [batch_size, seq_len]
    char_idx = char_idx.permute(1, 0)
    seq_len = char_idx.shape[1]
    # offset [batch,size, seq_lem]
    offset = torch.arange(0, batch_size * vocab_size, vocab_size, device=char_idx.device).unsqueeze(-1)
    char_idx_with_offset = char_idx + offset
    flatten_char_idx_with_offset = char_idx_with_offset.contiguous().view(-1)
    # [batch_size  * vocab_size, dim]
    flatten_embeddings = char_embeddings.contiguous().view(-1, dim)
    # selected embedding: [batch_size*seq_len, dim]
    flatten_selected_char_embedding = torch.index_select(flatten_embeddings, 0, flatten_char_idx_with_offset)
    selected_char_embedding = flatten_selected_char_embedding.view(batch_size, seq_len, -1)
    return selected_char_embedding.permute(1, 0, 2).contiguous()


def split_inputs(value, part_size):
    if torch.is_tensor(value):
        return [x.contiguous() for x in torch.split(value, part_size, -1)]
    elif isinstance(value, tuple):
        res = []
        for item in value:
            parts = split_inputs(item, part_size)
            for idx, part in enumerate(parts):
                if len(res) < idx + 1:
                    res.append([part])
                else:
                    res[idx].append(part)
        res = [tuple(x) for x in res]
        return res


def batch_to_dicts(batch, accumulation_steps=1):
    if accumulation_steps == 1:
        return [batch]
    batch_dict = dict()
    part_size = batch.batch_size // accumulation_steps
    for field in batch.fields:
        batch_field = getattr(batch, field)
        if torch.is_tensor(batch_field):
            batch_dict[field] = batch_field
        elif isinstance(batch_field, tuple):
            new_field = [batch_field[0], batch_field[1]]
            batch_dict[field] = tuple(new_field)
        else:
            batch_dict[field] = batch_field

    # 如果不是，开始分割
    res = [dict() for i in range(accumulation_steps)]
    for key, value in batch_dict.items():
        # 这个不能进行分batch
        if key.startswith('dmc_bdv_'):
            parts1 = split_inputs(value[0], part_size)
            parts2 = [copy.deepcopy(value[1]) for x in range(accumulation_steps)]
            for idx, (p1, p2) in enumerate(zip(parts1, parts2)):
                res[idx][key] = tuple([p1, p2])
        elif key.startswith('dmc_'):
            raise NotImplemented()
        else:
            parts = split_inputs(value, part_size)
            for idx, part in enumerate(parts):
                res[idx][key] = part
    return [ParamDict(x) for x in res]


def fix_parameters(params, model):
    """
       进行参数固定
    """
    if 'partial_training_rules' in params:
        rules = params.partial_training_rules
        for rule in rules:
            rule_type = rule[0]
            rule_value = rule[1]

            if rule_type == 'fix_by_prefix':
                for name, param in model.named_parameters():
                    if name.startswith(rule_value):
                        param.requires_grad = False
                        logger.info('[Fix Params]: [%s/%s] fixed %s' % (rule_type, rule_value, name))
            else:
                raise NotImplementedError()

    logger.info('[---------------------------Updated Params:')
    show_parameters(model)
