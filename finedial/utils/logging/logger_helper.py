# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import os
from logging.handlers import RotatingFileHandler

logger = None

def init_logger(log_path=None, experiment_name=None, run_name='run_log', log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if experiment_name and experiment_name != '' and log_path and log_path != '':
        log_dir = os.path.join(log_path, experiment_name)
        if not os.path.isdir(log_dir):
            logger.info('[CHECKPOINT] Creating log file folder:%s' % log_dir)
            os.makedirs(log_dir)
        log_file = os.path.join(log_path, experiment_name, '%s.log' % run_name)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=1000000, backupCount=10)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

init_logger()