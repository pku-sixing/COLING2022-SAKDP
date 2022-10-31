import argparse
import os
import time

from finedial.framework.runner.run_l2r_model import run_finel2r
from finedial.utils.logging.logger_helper import logger

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--config', type=str, default='configs/debug.json', help='model config')
parser.add_argument('--mode', type=str, default='default', help='default, train, eval, infer， infer_ensemble')
parser.add_argument('--wait_gpu', type=int, default=-1, help='if not -1, wait the GPU')
parser.add_argument('--infer_batch_size', type=int, default=10, help='if not -1, wait the GPU')

args = parser.parse_args()

readme = \
"""
README:
   In the DMC mode, we will not copy tokens whose vocabulary id is less than 100
"""

if __name__ == '__main__':
    print(readme)
    if args.wait_gpu != -1:
        first_try = True
        while True:
            command = os.popen("nvidia-smi -q -d PIDS | grep Processes")
            lines = command.read().split("\n")
            free_gpu = []
            busy_gpu = []
            for i in range(len(lines)):
                print(i, lines[i])
                if len(lines[i]) < 1:
                    continue
                if "None" in lines[i]:
                    free_gpu.append(i)
                else:
                    busy_gpu.append(i)
            logger.info('busy_gpu ' + str(busy_gpu))
            logger.info('free_gpu ' + str(free_gpu))
            if args.wait_gpu in free_gpu:
                logger.info(str(args.wait_gpu) + ' is available now')
                if not first_try:
                    logger.info(str(args.wait_gpu) + ' is available now, run in 30 mins')
                    time.sleep(60 * 10)
                break
            else:
                logger.info(str(args.wait_gpu) + ' is busy now')
                time.sleep(60 * 15)
                first_try = False
    run_finel2r(args, args.infer_batch_size)
    print(readme)

