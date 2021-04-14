# -*- coding: utf-8 -*-
'''
Created on 2021/02/27 18:13:39

@File -> settings.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 默认设置
'''

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../'))
sys.path.append(BASE_DIR)

from mod.config.config_loader import config_loader

PROJ_DIR, PROJ_CMAP = config_loader.proj_dir, config_loader.proj_cmap
proj_plt = config_loader.proj_plt


# 载入项目变量配置.
ENC_CONFIG = config_loader.environ_config
MODEL_CONFIG = config_loader.model_config
TEST_PARAMS = config_loader.test_params
