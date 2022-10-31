import time
from collections import defaultdict

import torch

from finedial.framework.model.BaseWrapper import BaseWrapper
from finedial.ops.general import model_helper
from finedial.utils.logging import progress_helper


class L2RTrainingWrapper(BaseWrapper):

    def __init__(self, model_dict, optimizer_dict, params_dict):
        """
            model_dict: 所有装载的模型
        """
        super().__init__(model_dict, optimizer_dict, params_dict)

    def run_epoch(self, epoch, batch_iter, is_eval=False, mask_eval_unk=False, default_accumulation_steps=1):
        if is_eval:
            self.eval()
        else:
            self.train()

        start_time = time.time()
        p_bar = progress_helper.get_progressbar(len(batch_iter))
        p_bar.start()
        loss_dict = defaultdict(int)
        global_loss_dict = defaultdict(float)
        instance_count = 0.0

        # 分批进行处理
        for b, batch_data in enumerate(batch_iter):
            p_bar.update(b)
            # Run A Step
            raw_batch_size = batch_data.batch_size
            accumulation_steps = default_accumulation_steps
            while True:
                try:
                    loss_for_bp_dicts = []
                    local_loss_dict_dicts = []
                    batch_dicts = model_helper.batch_to_dicts(batch_data, accumulation_steps=accumulation_steps)
                    for batch_dict in batch_dicts:
                        loss_for_bp_dict, local_loss_dict_dict = self.forward(batch_dict, epoch)
                        loss_for_bp_dicts.append(loss_for_bp_dict)
                        local_loss_dict_dicts.append(local_loss_dict_dict)
                        for model_name, loss_for_bp in loss_for_bp_dict.items():
                            loss_for_bp = loss_for_bp.mean()
                            if not is_eval:
                                # Optimizing Step
                                loss_for_bp = loss_for_bp / accumulation_steps
                                loss_for_bp.backward()
                    # 正常执行完成了，需要退出
                    break
                except RuntimeError as re:
                    for model_name, loss_for_bp in loss_for_bp_dict.items():
                        optimizer = self.optimizer_dict[model_name]
                        optimizer.zero_grad()
                    del loss_for_bp_dicts
                    del local_loss_dict_dicts
                    del batch_dicts
                    print(
                        'mem_over_flow, decrease batch size from  %d to %d \n' % (raw_batch_size // accumulation_steps,
                                                                                  raw_batch_size // accumulation_steps // 2))
                    accumulation_steps *= 2
                    if accumulation_steps > raw_batch_size:
                        raise ValueError()

            for loss_for_bp_dict, local_loss_dict_dict in zip(loss_for_bp_dicts, local_loss_dict_dicts):
                for model_name, loss_for_bp in loss_for_bp_dict.items():
                    loss_for_bp = loss_for_bp.mean()
                    local_loss_dict = local_loss_dict_dict[model_name]
                    params = self.params_dict[model_name]
                    loss_terms = params.training['loss_terms']
                    report_step_intervals = params.training['report_step_intervals']

                    if 'contrastive_loss' in loss_terms:
                        loss = local_loss_dict['contrastive_loss'].mean().item() * 100
                        loss_dict['contrastive_loss'] += (loss / accumulation_steps)
                        # TGT Len
                        instance_count += len(batch_dict.tgt[1])
                        global_loss_dict['contrastive_loss'] += loss * len(batch_dict.tgt[1])
                    global_loss_dict['loss'] += loss_for_bp.item() * len(batch_dict.tgt[1])

                    # multi_stage
                    for loss_term in ['pos_neg', 'pos_neg2', 'neg_neg2']:
                        if loss_term in local_loss_dict:
                            loss = local_loss_dict[loss_term].mean().item() * 100
                            loss_dict[loss_term] += (loss / accumulation_steps)
                            # TGT Len
                            global_loss_dict[loss_term] += loss * len(batch_dict.tgt[1])

            if not is_eval:
                for model_name, loss_for_bp in loss_for_bp_dict.items():
                    optimizer = self.optimizer_dict[model_name]
                    model = self.model_dict[model_name]
                    params = self.params_dict[model_name]
                    torch.nn.utils.clip_grad_norm_(model.parameters(), params.training['grad_clip'])
                    optimizer.step()
                    optimizer.zero_grad()

                # Report Step
                if b % report_step_intervals == 0 and b != 0:
                    time_diff = time.time() - start_time
                    progress_helper.step_report(epoch, b, report_step_intervals, len(batch_iter),
                                                loss_dict, time_diff, optimizer)
                    loss_dict = defaultdict(int)

        p_bar.finish()

        for key in global_loss_dict:
            if key != 'mse_word' and key != 'word_cnt':
                global_loss_dict[key] = global_loss_dict[key] / instance_count

        res_dict = {'global_loss': global_loss_dict}
        return res_dict

    def forward(self, batch_data, epoch_num):
        """
            Forward Step
        """
        # 首先获得输入
        inputs_dict = {}
        for model_name, model in self.model_dict.items():
            inputs_dict[model_name] = model.get_inputs_from_batch(batch_data, infer_mode=False)
        # 进行Encoder的编码
        global_loss_for_bp = dict()
        global_loss_dict = dict()
        for model_name, model in self.model_dict.items():
            # Encoding， 均化长度
            contrastive_loss, batch_loss_dict, normalization_factor = model.batch_level_train(batch_data, epoch_num)
            loss_for_bp = 0
            local_loss_dict = defaultdict(float)
            if 'contrastive_loss' in model.loss_terms:
                loss_for_bp += contrastive_loss
                local_loss_dict['contrastive_loss'] = contrastive_loss.detach() / normalization_factor
                for loss_key, loss_val in batch_loss_dict.items():
                    local_loss_dict[loss_key] = loss_val.detach() / normalization_factor
            global_loss_for_bp[model_name] = loss_for_bp.view(1, -1)
            global_loss_dict[model_name] = local_loss_dict
        return global_loss_for_bp, global_loss_dict

    def infer(self, batch_data):
        """
            Forward Step
        """
        # 首先获得输入
        inputs_dict = {}
        for model_name, model in self.model_dict.items():
            inputs_dict[model_name] = model.get_inputs_from_batch(batch_data, infer_mode=False)
        # 进行Encoder的编码
        global_loss_for_bp = dict()
        global_loss_dict = dict()

        for model_name, model in self.model_dict.items():
            # Encoding， 均化长度
            scores, hybrid_scores = model.infer_batch(batch_data)
            res_dict = defaultdict(float)
            res_dict['pos_scores'] = scores.detach()
            global_loss_dict[model_name] = res_dict

        return global_loss_for_bp, global_loss_dict
