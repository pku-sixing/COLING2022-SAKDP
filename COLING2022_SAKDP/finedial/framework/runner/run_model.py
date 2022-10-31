import json
import os
import pprint
import time
from collections import OrderedDict

import numpy as np
import torch
from tensorboardX import SummaryWriter

from finedial.framework.model.EnsembleSeq2Seq import EnsembleSeq2Seq
from finedial.framework.model.InferenceWrapper import InferenceWrapper
from finedial.framework.model.TrainingWrapper import TrainingWrapper
from finedial.framework.model.create_model import create_model
from finedial.ops.general import model_helper
from finedial.ops.general.model_helper import get_optimizer_fn
from finedial.utils.data_utils import load_dataset, param_helper
from finedial.utils.evaluation import generation_helper
from finedial.utils.evaluation.score_helper import ScoreManager
from finedial.utils.logging import logger_helper
from finedial.utils.logging import progress_helper
from finedial.utils.logging.logger_helper import logger


def infer_epoch(params, model, batch_iter, field_dict, word_sample_mode, infer_set='test'):
    model.eval()
    tgt_field = field_dict['tgt']
    dmf_vocabs = field_dict['dmf_vocab'][infer_set]
    mode_name = word_sample_mode['mode_name']
    logger.info("[EPOCH] Running a new infer epoch: %s" % mode_name)
    logger.info(pprint.pformat(word_sample_mode))
    ignore_token_set = {'<ssos>', '<sos>'}
    eos_token_set = {'<seos>', '<eos>'}
    references_order = params.dataset.copy_order
    logger.info("[EPOCH] reference order: %s" % '->'.join(references_order))
    p_bar = progress_helper.get_progressbar(len(batch_iter))
    p_bar.start()
    generations = []
    meta_generations = []
    case_id = 0
    beam_width = word_sample_mode['beam_width']
    for b, batch in enumerate(batch_iter):
        p_bar.update(b)
        output_dict = model.infer(batch, word_sample_mode=word_sample_mode, infer_length=params.dataset.max_tgt_len)
        generated_responses = output_dict['generated_responses'].cpu().numpy()
        generated_scores = output_dict['generated_scores'].cpu().numpy()
        batch_size_mul_beam = generated_scores.shape[0]
        batch_size = batch_size_mul_beam // beam_width
        generated_responses = generated_responses.reshape([batch_size, beam_width, -1])
        generated_scores = generated_scores.reshape([batch_size, beam_width, -1])

        def translate(response, case_id, local_vocab):
            generation = []
            for tid in response:
                token = local_vocab[tid.item()]
                if len(dmf_vocabs) > 0:
                    if token in dmf_vocabs[case_id]:
                        token = dmf_vocabs[case_id].get(token)
                    else:
                        pass
                if token in ignore_token_set:
                    continue
                elif token not in eos_token_set:
                    generation.append(token)
                else:
                    break
            return ' '.join(generation)

        # Inputs
        if 'dmc_query' in batch.__dict__:
            src = batch.dmc_query.transpose(0, 1)
        else:
            if 'src' in batch:
                src = batch.src[0].transpose(0, 1)
            else:
                src = None
        tgt = batch.tgt[0].transpose(0, 1)

        reference_inputs = []
        for reference in references_order:
            if 'dmc_' + reference in batch.__dict__:
                tmp_itr = batch.__dict__['dmc_' + reference]
                reference_data = tmp_itr.transpose(0, 1)
            elif 'dmc_bdv_' + reference in batch.__dict__:
                tmp_itr = batch.__dict__['dmc_bdv_' + reference]
                ptos_dict = dict()
                for k, v in tmp_itr[1][0].items():
                    ptos_dict[v] = k
                reference_data = (tmp_itr[0].transpose(0, 1), ptos_dict)
            else:
                if reference == 'query':
                    reference = 'src'
                tmp_itr = batch.__dict__[reference]
                reference_data = tmp_itr[0]
                reference_data = reference_data.transpose(0, 1)
            # if reference == 'query':
            #     continue
            reference_inputs.append((reference, reference_data))

        for batch_id in range(0, batch_size):
            meta_generation = OrderedDict()
            scores = generated_scores[batch_id].sum(-1)
            indices = np.argsort(-scores)
            # Meta Generations
            if src is not None:
                meta_generation['query'] = translate(src[batch_id], case_id, field_dict['src'].vocab.itos)
            meta_generation['response'] = translate(tgt[batch_id], case_id, field_dict['tgt'].vocab.itos)
            for reference_input in reference_inputs:
                if 'dmc_' + reference_input[0] in field_dict:
                    meta_generation[reference_input[0]] = translate(reference_input[1][batch_id], case_id,
                                                                    field_dict['dmc_' + reference_input[0]].vocab.itos)
                elif 'dmc_bdv_' + reference_input[0] in field_dict:
                    meta_generation[reference_input[0]] = translate(reference_input[1][0][batch_id], case_id,
                                                                    reference_input[1][1])
            if len(dmf_vocabs) > 0:
                tmp = []
                for key in dmf_vocabs[case_id]:
                    value = dmf_vocabs[case_id][key]
                    tmp.append((key, value))
                sorted_items = sorted(tmp, key=lambda x: x[1])
                for idx in range(0, len(sorted_items), 10):
                    sub_tmp = sorted_items[idx:idx + 10]
                    vocab_str = ['%s-%s' % x for x in sub_tmp]
                    vocab_str = ' '.join(vocab_str)
                    meta_generation['dmc-vocab %s-%s' % (idx, idx + 10)] = vocab_str
            for beam_id in range(0, beam_width):
                beam_index = indices[beam_id]
                response = generated_responses[batch_id][beam_index]
                score = generated_scores[batch_id][beam_index]
                generation = translate(response, case_id, tgt_field.vocab.itos)
                generations.append(generation)
                meta_generation['generation_%d_%.2f' % (beam_id, score.sum())] = generation
            meta_generations.append(meta_generation)
            case_id += 1

    vocab_size = len(tgt_field.vocab.stoi)
    res_dict = generation_helper.write_generation_results(params, generations, meta_generations, vocab_size,
                                                          mode_name, beam_width,
                                                          word_sample_mode.get("ref_file_suffix", ""))
    p_bar.finish()
    return res_dict


def infer(params):
    states = {}
    batch_size = params.infer.get("batch_size", None)
    train_iter, val_iter, test_iter, field_dict, dialogue_sp_token_dicts = load_dataset.load_dataset(params, is_eval=True, dmf_vocab=True,
                                                                            batch_size=batch_size)
    # Load model
    model = create_model(params, dialogue_sp_token_dicts)
    assert params.eval['use_best_model']
    model_helper.try_restore_model(params.model_path, params.experiment_name, model, None, states,
                                   best_model=params.eval['use_best_model'], cuda=params.cuda)
    if params.cuda is False:
        model = model.cpu()
    logger.info(pprint.pformat(states))
    logger.info(model)
    trainer = InferenceWrapper(model, params)

    word_sample_modes = params.infer['word_sample_modes']
    score_manager = ScoreManager(result_path=params.model_path, experiment_name=params.experiment_name)

    for idx, word_sample_mode in enumerate(word_sample_modes):
        if word_sample_mode.get("ignore", False):
            logger.info('[Decoding] %d/%d, Ignoring Sample Mode:%s ' % (idx, len(word_sample_modes),
                                                                        word_sample_mode['mode_name']))
            continue
        logger.info('[Decoding] %d/%d, Sample Mode:%s ' % (idx, len(word_sample_modes), word_sample_mode['mode_name']))
        res_dict = infer_epoch(params, trainer, test_iter, field_dict, word_sample_mode,
                               infer_set='test')
        score_manager.update_group(word_sample_mode['mode_name'], res_dict)


def infer_ensemble(ensemble_params):
    states = {}
    major_params = ensemble_params['major_model']
    batch_size = major_params.infer.get("batch_size", None)
    train_iter, val_iter, test_iter, field_dict, dialogue_sp_token_dicts = load_dataset.load_dataset(major_params, is_eval=True, dmf_vocab=True,
                                                                            batch_size=batch_size)
    # Load model
    model_dict = {}
    for model_name, params in ensemble_params.items():
        logger.info('[Ensemble-Model] Loading %s' % model_name)
        model = create_model(params, dialogue_sp_token_dicts)
        model_helper.try_restore_model(params.model_path, params.experiment_name, model, None, states,
                                       best_model=params.eval['use_best_model'], cuda=params.cuda)
        if params.cuda is False:
            model = model.cpu()
        logger.info(pprint.pformat(states))
        logger.info(model)

        model_dict[model_name] = model

    word_sample_modes = major_params.infer['word_sample_modes']
    score_manager = ScoreManager(result_path=major_params.model_path, experiment_name=major_params.experiment_name)
    model = EnsembleSeq2Seq(model_dict)

    for idx, word_sample_mode in enumerate(word_sample_modes):
        if word_sample_mode.get("ignore", False):
            logger.info('[Decoding] %d/%d, Ignoring Sample Mode:%s ' % (idx, len(word_sample_modes),
                                                                        word_sample_mode['mode_name']))
            continue
        logger.info('[Decoding] %d/%d, Sample Mode:%s ' % (idx, len(word_sample_modes), word_sample_mode['mode_name']))
        res_dict = infer_epoch(major_params, model, test_iter, field_dict, word_sample_mode,
                               infer_set='test')
        score_manager.update_group(word_sample_mode['mode_name'], res_dict)


def eval(params, output_score=False):
    states = {}
    _, val_iter, test_iter, field_dict, dialogue_sp_token_dicts = load_dataset.load_dataset(params, is_eval=True)
    # Load model
    model = create_model(params, dialogue_sp_token_dicts)
    assert params.eval['use_best_model']
    model_helper.try_restore_model(params.model_path, params.experiment_name, model, None, states,
                                   best_model=params.eval['use_best_model'], cuda=params.cuda)
    logger.info(pprint.pformat(states))
    trainer = TrainingWrapper(model, None, params)
    # Score Manager
    score_manager = ScoreManager(result_path=params.model_path, experiment_name=params.experiment_name)
    if output_score is False:
        res = trainer.run_epoch(-1, val_iter, is_eval=True)
        score_manager.update_group("eval_on_validation", res)
        logger.info('[Scores] Eval results on Validation')
        logger.info(pprint.pformat(res))
    res = trainer.run_epoch(-1, test_iter, is_eval=True, output_score=output_score)
    if output_score:
        score_manager.update_group("eval_on_test", res['global_loss'])
    logger.info('[Scores] Eval results on Test')
    logger.info(pprint.pformat(res))
    return res


def train(params, config_path):
    states = {'val_loss': [], 'val_ppl': [], 'lr': params.training['learning_rate'], 'epoch': 0, 'best_epoch': 0,
              'best_val_loss': -1}

    record_writer = SummaryWriter(os.path.join(params.model_path, params.experiment_name, 'tensorboard'))
    train_iter, val_iter, test_iter, field_dict, dialogue_sp_token_dicts = load_dataset.load_dataset(params)

    # Load model
    model = create_model(params, dialogue_sp_token_dicts)
    optimizer_name = params.training.get('optimizer', 'Adam')
    logger.info('[Optimizer]=%s' % optimizer_name)
    optimizer_fn = get_optimizer_fn(optimizer_name)
    if params.training.get('optimizer_group', None) is None:
        optimizer = optimizer_fn(model.parameters(), lr=params.training['learning_rate'])
    else:
        groups = params.training.get('optimizer_group')
        default_param = {
            'params': []
        }
        group_params = []
        for group in groups:
            group_params.append(
                {
                    'params': [],
                    'lr': group.learning_rate
                }
            )
        for p_name, p in model.named_parameters():
            default = True
            for gidx, group in enumerate(groups):
                if p_name.startswith(group.prefix):
                    group_params[gidx]['params'].append(p)
                    logger.info('[Optimizer Group]  %s-%s' % (group.name, p_name ))
                    default = False
                    break
            if default:
                default_param['params'].append(p)
                logger.info('[Optimizer Group]  %s-%s' % ('default', p_name))

        optimizer = optimizer_fn([default_param] + group_params, lr=params.training['learning_rate'])

    model_helper.try_restore_or_load_model(params, field_dict, config_path, model, optimizer, states,
                                           best_model=False)
    # Fixed Parameters
    model_helper.fix_parameters(params, model)

    start_epoch = states['epoch']
    logger.info(model)

    trainer = TrainingWrapper(model, optimizer, params)
    eval_before_training = params.training.get('eval_before_training', False)
    eval_before_training_offset = 1 if eval_before_training else 0

    for epoch in range(start_epoch + 1, params.training['epochs'] + 1 + eval_before_training_offset):
        if epoch - states['best_epoch'] > 2 and params.training['early_stop'] == 'general':
            logger.info(' Best Epoch :%d, Current Epoch %d, Max Epoch %d' % (states['best_epoch'], epoch,
                                                                             params.training['epochs']))
            logger.info('[STOP] Best Epoch :%d, Current Epoch %d, Max Epoch %d' % (states['best_epoch'],
                                                                                   epoch, params.training['epochs']))
            return False
        start_time = time.time()
        num_batches = len(train_iter)
        logger.info('[NEW EPOCH] %d/%d, num of batches : %d' % (epoch, params.training['epochs'], num_batches))
        # Train Step
        if epoch == 1 and eval_before_training:
            logger.info('[NEW EPOCH] %d/%d, SKIP because eval_before_training, num of batches : %d' % (epoch, params.training['epochs'], num_batches))
            pass
        else:
            trainer.run_epoch(epoch, train_iter,
                              default_accumulation_steps=params.dataset.get('default_accumulation_steps', 1))
        # Eval on Val & Test
        eval_res_val = trainer.run_epoch(epoch, val_iter, is_eval=True)
        eval_res_test = trainer.run_epoch(epoch, test_iter, is_eval=True)

        val_ppl = eval_res_val['global_loss']['overall_ppl']
        val_loss = eval_res_val['global_loss']['mse']
        test_ppl = eval_res_test['global_loss']['overall_ppl']
        test_loss = eval_res_test['global_loss']['mse']
        test_word_ppx = eval_res_test['global_loss'].get('word_ppx', -1)
        val_word_ppx = eval_res_val['global_loss'].get('word_ppx', -1)

        record_writer.add_scalar('test/ppl', test_ppl, epoch)
        record_writer.add_scalar('test/loss', test_loss, epoch)
        record_writer.add_scalar('val/ppl', val_ppl, epoch)
        record_writer.add_scalar('val/loss', val_loss, epoch)

        logger.info("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2f/%5.2f | test_loss:%5.3f | test_pp:%5.2f/%5.2f"
                    % (epoch, val_loss, val_ppl, val_word_ppx, test_loss, test_ppl, test_word_ppx))
        time_diff = (time.time() - start_time) / 3600.0
        logger.info("[Epoch:%d] epoch time:%.2fH, est. remaining  time:%.2fH" %
                    (epoch, time_diff, time_diff * (params.training['epochs'] - epoch)))

        # Adjusting learning rate
        states['val_ppl'].append(val_ppl)
        states['val_loss'].append(val_loss)
        states['epoch'] = epoch
        if len(states['val_ppl']) >= 2:
            logger.info('[TRAINING] last->now valid ppl : %.3f->%.3f' % (states['val_ppl'][-2], states['val_ppl'][-1]))

            if epoch >= params.training['start_to_adjust_lr'] and states['val_ppl'][-1] >= \
                    states['val_ppl'][-2]:
                logger.info('[TRAINING] Adjusting learning rate due to the increment of val_loss')
                new_lr = model_helper.adjust_learning_rate(optimizer, rate=params.training['lr_decay_rate'])
                states['lr'] = new_lr

        # Save the model if the validation loss is the best we've seen so far.
        if states['best_val_loss'] == -1 or val_ppl < states['best_val_loss']:
            logger.info('[CHECKPOINT] New best valid ppl : %.3f->%.2f' % (states['best_val_loss'], val_ppl))
            model_helper.save_model(params.model_path, params.experiment_name, epoch, val_ppl, model, optimizer,
                                    params, states, best_model=True, clear_history=True)
            states['best_val_loss'] = val_ppl
            states['best_epoch'] = epoch

        # Saving standard model
        model_helper.save_model(params.model_path, params.experiment_name, epoch, val_ppl, model, optimizer,
                                params, states, best_model=False, clear_history=True)

        if params.training.get('reload_and_shuffle_each_epoch', False):
            # 如果不是最后一个就范围
            if epoch != params.training['epochs'] + eval_before_training_offset:
                model_helper.set_seed(params.random_seed+epoch)
                logger.info('[EPOCH] Reload model, and reset random seed to %d' % (params.random_seed+epoch))
                return True
    return False



def run_finedial(args):
    params = param_helper.ParamDict(params=json.load(open(args.config, encoding='utf-8')))

    if params.cuda:
        logger.info('[PARAM] Enabling CUDA')
        if not torch.cuda.is_available():
            logger.info('[PARAM] CUDA is not available now, the model will run on CPU after 1min')
            params.cuda = False


    if args.mode == 'infer_ensemble':
        major_params = params['major_model']
        all_params = list(params.values())
    else:
        major_params = params
        all_params = [params]

    logger_helper.init_logger(major_params.get('log_path', None), major_params.get('experiment_name', None),
                              run_name=args.mode)
    logger.info('Mode: %s, Configs:' % args.mode)
    logger.info(json.dumps(major_params, indent=1, ensure_ascii=False))
    model_helper.set_seed(major_params.random_seed)

    # 为了获得准确的词表大小
    a, b, c, field_dict, dialogue_sp_token_dicts = load_dataset.load_dataset(major_params, is_eval=True)
    fixed_vocab_size = len(field_dict['tgt'].vocab)
    for idx in range(0, len(field_dict['tgt'].vocab)):
        if field_dict['tgt'].vocab.itos[idx] == '<dmc_0>':
            fixed_vocab_size = idx
            break

    extend_vocab_size = len(field_dict['tgt'].vocab)
    logger.info('[VOCAB_SIZE] FIXED: %d, FIXED+PLACEHOLDER: %d' % (fixed_vocab_size, extend_vocab_size))
    if 'placeholder_num' in major_params.dataset:
        extend_vocab_size = fixed_vocab_size + major_params.dataset.placeholder_num
        logger.info('[VOCAB_SIZE] FIXED: %d, (Overwrite) FIXED+PLACEHOLDER: %d' % (fixed_vocab_size, extend_vocab_size))

    logger.info('[VOCAB_SIZE] FIXED: %d, FIXED+PLACEHOLDER: %d' % (fixed_vocab_size, extend_vocab_size))

    for current_params in all_params:
        current_params.embedding['token_vocab_size'] = extend_vocab_size  # fixed_vocab_size
        current_params.dataset['token_output_size'] = extend_vocab_size
        current_params.stepwise_decoder['fixed_vocab_size'] = fixed_vocab_size
        current_params.stepwise_decoder['extend_vocab_size'] = extend_vocab_size - fixed_vocab_size

        if 'references' in params:
            for reference in current_params.references:
                if reference.knowledge_type == 'text' or reference.knowledge_type == 'sub_text':
                    logger.info(
                        '[VOCAB_SIZE] Local Vocab Size of %s %d' % (
                            reference.name, len(field_dict[reference.name].vocab)))
                    reference.embedding.token_vocab_size = len(field_dict[reference.name].vocab)
                elif reference.knowledge_type == 'svo':
                    logger.info(
                        '[VOCAB_SIZE] Local Vocab Size of %s %d' % (
                            reference.name, len(field_dict[reference.name + '_head'].vocab)))
                    reference.embedding.token_vocab_size = len(field_dict[reference.name + '_head'].vocab)
                elif reference.knowledge_type == 'infobox':
                    logger.info(
                        '[VOCAB_SIZE] Local Vocab Size of %s %d' % (
                            reference.name, len(field_dict[reference.name + '_key'].vocab)))
                    reference.embedding.token_vocab_size = len(field_dict[reference.name + '_key'].vocab)

    if args.mode == 'train':
        while True:
            res = train(params, args.config)
            if res is False:
                break
    elif args.mode == 'eval':
        eval(params)
    elif args.mode == 'eval_score':
        eval(params, output_score=True)
    elif args.mode == 'infer':
        infer(params)
    elif args.mode == 'infer_ensemble':
        infer_ensemble(params)
    elif args.mode == 'default':
        while True:
            res = train(params, args.config)
            if res is False:
                break
        eval(params)
        infer(params)
