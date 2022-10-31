"""
实现训练进度条的
"""
import math

import progressbar
from finedial.utils.logging.logger_helper import logger

def get_progressbar(max_step=100):
    widgets = ['Progress:', progressbar.Percentage(),' ', progressbar.Bar('🐱'), progressbar.Timer(),
               ' ', progressbar.ETA(), ' ', progressbar.FileTransferSpeed()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=max_step, term_width=74)
    return pbar

def step_report(epoch, step, interval, total_step, loss_dict, time_diff, optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(str(param_group['lr']))
    lrs = ','.join(lrs)
    time_per_step = time_diff / step
    remaining_time = time_per_step * (total_step - step)
    # 显示最准确的Loss
    tmp = []
    for key in loss_dict:
        tmp.append('%s:%.2f' % (key, loss_dict[key] / interval))
    ppl_loss = loss_dict['ppl_loss'] / interval
    logger.info(
        "[Epoch=%d/Step=%d-%d][lrs:%s][%s][ppl:%5.2f][step_time:%.2fs][remain:%.2fs]" %
        (epoch, step, total_step, lrs, ','.join(tmp), math.exp(ppl_loss), time_per_step, remaining_time))