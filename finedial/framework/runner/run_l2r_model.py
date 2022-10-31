import json
import os
import pprint
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter

from finedial.framework.model.learning2rank.L2RTrainingWrapper import L2RTrainingWrapper
from finedial.framework.model.learning2rank.create_l2r_model import create_l2r_model
from finedial.ops.general import model_helper
from finedial.ops.general.model_helper import get_optimizer_fn
from finedial.utils.data_utils import load_dataset, param_helper
from finedial.utils.evaluation.score_helper import ScoreManager
from finedial.utils.logging import logger_helper, progress_helper
from finedial.utils.logging.logger_helper import logger


def eval(params):
    states = {}
    _, val_iter, test_iter, field_dict, dialogue_sp_token_dicts = load_dataset.load_dataset(params, is_eval=True)
    # Load model
    model = create_l2r_model(params, dialogue_sp_token_dicts)
    assert params.eval['use_best_model']
    model_helper.try_restore_model(params.model_path, params.experiment_name, model, None, states,
                                   best_model=params.eval['use_best_model'], cuda=params.cuda)
    logger.info(pprint.pformat(states))
    trainer = L2RTrainingWrapper(model, None, params)
    # Score Manager
    score_manager = ScoreManager(result_path=params.model_path, experiment_name=params.experiment_name)
    res = trainer.run_epoch(-1, val_iter, is_eval=True)
    score_manager.update_group("eval_on_validation", res)
    logger.info('[Scores] Eval results on Validation')
    logger.info(pprint.pformat(res))
    res = trainer.run_epoch(-1, test_iter, is_eval=True)
    score_manager.update_group("eval_on_test", res)
    logger.info('[Scores] Eval results on Test')
    logger.info(pprint.pformat(res))
    return res


def infer_epoch(params, model, batch_iter, field_dict):
    model.eval()
    mode_name = "general prediction"
    logger.info("[EPOCH] Running a new infer epoch: %s" % mode_name)
    logger.info(pprint.pformat(mode_name))
    references_order = params.dataset.copy_order
    logger.info("[EPOCH] reference order: %s" % '->'.join(references_order))
    p_bar = progress_helper.get_progressbar(len(batch_iter))
    p_bar.start()
    pos_scores = []
    hybrid_scores = []
    for b, batch in enumerate(batch_iter):
        p_bar.update(b)
        res_dict = model.infer(batch)
        pos_score = res_dict[1]['major_model']['pos_scores'].cpu()
        for score in pos_score.tolist():
            pos_scores.append(score)

        if 'hybrid_scores' in res_dict[1]['major_model']:
            hybrid_score = res_dict[1]['major_model']['hybrid_scores'].cpu()
            for score in hybrid_score.tolist():
                hybrid_scores.append(score)

    p_bar.finish()
    return pos_scores, hybrid_scores


def infer(params):
    states = {}
    batch_size = params.infer.get("batch_size", None)
    train_iter, val_iter, test_iter, field_dict, dialogue_sp_token_dicts = load_dataset.load_dataset(params,
                                                                                                     is_eval=True,
                                                                                                     dmf_vocab=True,
                                                                                                     batch_size=batch_size)
    # Load model
    model = create_l2r_model(params, dialogue_sp_token_dicts)
    assert params.eval['use_best_model']
    model_helper.try_restore_model(params.model_path, params.experiment_name, model, None, states,
                                   best_model=params.eval['use_best_model'], cuda=params.cuda)
    if params.cuda is False:
        model = model.cpu()
    logger.info(pprint.pformat(states))
    logger.info(model)
    trainer = L2RTrainingWrapper(model, None, params)
    pos_scores, hybrid_scores = infer_epoch(params, trainer, test_iter, field_dict)
    result_path = os.path.join(params.model_path, params.experiment_name)
    with open(result_path + '/rank_scores%s.txt' % (params.dataset.get("result_suffix", "")), 'w+', encoding='utf-8') as fout:
        for score in pos_scores:
            fout.write('%.4f\n' % score)

    if len(hybrid_scores) > 0:
        # Compute Hit
        hit_array = np.zeros([len(hybrid_scores[0])])
        mrr = 0
        hits = []

        turn_id = 0
        for pos_score, hybrid_score in zip(pos_scores, hybrid_scores):
            try:
                # 必须这么做，否则精度会出问题
                hybrid_score[turn_id % batch_size] = pos_score
                sorted_score = sorted(hybrid_score, reverse=True)
                hit = 0
                while sorted_score[hit] > pos_score and hit < len(sorted_score) - 1:
                    hit = hit + 1
                mrr += 1 / (hit + 1)
                for idx in range(hit, len(hybrid_scores[0])):
                    hit_array[idx] += 1
                hit = hit + 1
                hits.append(hit)
                assert hit <= len(sorted_score)
                turn_id += 1
            except Exception as e:
                print(pos_score)
                print(hybrid_score)
                raise e
        with open(result_path + '/rank_matrix_scores.txt', 'w+', encoding='utf-8') as fout:
            for hit, score in zip(hits, hybrid_scores):
                text = ['%.4f' % x for x in score]
                fout.write('%d\t%s\n' % (hit,' '.join(text)))

        score_manager = ScoreManager(result_path=params.model_path, experiment_name=params.experiment_name)
        hit_array = hit_array / len(hybrid_scores)
        res_dict = dict()
        res_dict['MRR'] = mrr / len(hybrid_scores)
        for idx, hit_rate in enumerate(hit_array):
            metric = 'recall_%d@%d' % (idx + 1, len(hybrid_scores[0]))
            res_dict[metric] = hit_rate
        score_manager.update_group("ranking", res_dict)


def train(params, config_path):
    states = {'val_loss': [], 'val_ppl': [], 'lr': params.training['learning_rate'], 'epoch': 0, 'best_epoch': 0,
              'best_val_loss': -1}

    record_writer = SummaryWriter(os.path.join(params.model_path, params.experiment_name, 'tensorboard'))
    train_iter, val_iter, test_iter, field_dict, dialogue_sp_token_dicts = load_dataset.load_dataset(params)

    # Load model
    model = create_l2r_model(params, dialogue_sp_token_dicts)
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
                    logger.info('[Optimizer Group]  %s-%s' % (group.name, p_name))
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

    trainer = L2RTrainingWrapper(model, optimizer, params)
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
            logger.info('[NEW EPOCH] %d/%d, SKIP because eval_before_training, num of batches : %d' % (
                epoch, params.training['epochs'], num_batches))
            pass
        else:
            trainer.run_epoch(epoch, train_iter,
                              default_accumulation_steps=params.dataset.get('default_accumulation_steps', 1))
        # Eval on Val & Test
        eval_res_val = trainer.run_epoch(epoch, val_iter, is_eval=True)
        eval_res_test = trainer.run_epoch(epoch, test_iter, is_eval=True)

        val_loss = eval_res_val['global_loss']['contrastive_loss']
        test_loss = eval_res_test['global_loss']['contrastive_loss']

        record_writer.add_scalar('test/contrastive_loss', test_loss, epoch)
        record_writer.add_scalar('val/contrastive_loss', val_loss, epoch)

        logger.info("[Epoch:%d] val_loss:%5.3f | test_loss:%5.3f "
                    % (epoch, val_loss, test_loss))
        time_diff = (time.time() - start_time) / 3600.0
        logger.info("[Epoch:%d] epoch time:%.2fH, est. remaining  time:%.2fH" %
                    (epoch, time_diff, time_diff * (params.training['epochs'] - epoch)))

        # Adjusting learning rate
        states['val_loss'].append(val_loss)
        states['epoch'] = epoch
        if len(states['val_loss']) >= 2:
            logger.info(
                '[TRAINING] last->now valid loss : %.3f->%.3f' % (states['val_loss'][-2], states['val_loss'][-1]))

            if epoch >= params.training['start_to_adjust_lr'] and states['val_loss'][-1] >= \
                    states['val_loss'][-2]:
                logger.info('[TRAINING] Adjusting learning rate due to the increment of val_loss')
                new_lr = model_helper.adjust_learning_rate(optimizer, rate=params.training['lr_decay_rate'])
                states['lr'] = new_lr

        # Save the model if the validation loss is the best we've seen so far.
        if states['best_val_loss'] == -1 or val_loss < states['best_val_loss']:
            logger.info('[CHECKPOINT] New best valid ppl : %.3f->%.2f' % (states['best_val_loss'], val_loss))
            model_helper.save_model(params.model_path, params.experiment_name, epoch, val_loss, model, optimizer,
                                    params, states, best_model=True, clear_history=True)
            states['best_val_loss'] = val_loss
            states['best_epoch'] = epoch

        # Saving standard model
        model_helper.save_model(params.model_path, params.experiment_name, epoch, val_loss, model, optimizer,
                                params, states, best_model=False, clear_history=True)
    return False


def run_finel2r(args, infer_batch_size):
    params = param_helper.ParamDict(params=json.load(open(args.config, encoding='utf-8')))
    params.infer.batch_size = infer_batch_size
    if params.cuda:
        logger.info('[PARAM] Enabling CUDA')
        if not torch.cuda.is_available():
            logger.info('[PARAM] CUDA is not available now, the model will run on CPU after 1min')
            params.cuda = False

    if args.mode == 'infer_ensemble':
        major_params = params['major_model']
    else:
        major_params = params

    logger_helper.init_logger(major_params.get('log_path', None), major_params.get('experiment_name', None),
                              run_name=args.mode)
    logger.info('Mode: %s, Configs:' % args.mode)
    logger.info(json.dumps(major_params, indent=1, ensure_ascii=False))
    model_helper.set_seed(major_params.random_seed)

    if args.mode == 'train':
        while True:
            res = train(params, args.config)
            if res is False:
                break
    elif args.mode == 'infer':
        infer(params)
    elif args.mode == 'infer_bulks':
        bulks = params.dataset.bulks
        st_bulk = bulks[0]
        end_bulk = bulks[1]
        original_query_suffix = params.dataset.query_suffix
        original_response_suffix = params.dataset.response_suffix
        original_result_suffix = params.dataset.result_suffix
        for bulk_id in range(st_bulk, end_bulk):
            try:
                print('bulkL: %d/%d-%d' % (bulk_id, st_bulk, end_bulk))
                params.dataset.query_suffix = original_query_suffix + ('_%d' % bulk_id)
                params.dataset.response_suffix = original_response_suffix + ('_%d' % bulk_id)
                params.dataset.result_suffix = original_result_suffix + ('_%d' % bulk_id)
                infer(params)
            except Exception as e:
                print(e)
    elif args.mode == 'default':
        while True:
            res = train(params, args.config)
            if res is False:
                break
        infer(params)

