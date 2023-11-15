# coding: utf-8
import os
import yaml
import socket
import time
import sys
import logging
import logging.config
import lake

import lake.decorator
import lake.data
import lake.dir

sys.path.append(os.path.join(os.path.dirname(__file__), '../config/'))

if len(logging.getLogger().handlers) == 0:
	logging.basicConfig(level=logging.DEBUG)

@lake.decorator.singleton
class ConfigLoader(object):
	def __init__(self, config_path=None):
		self._config_path = config_path or self._absolute_path('../config/config.yaml') # 日志配置文件
		self._load() # 载入日志配置文件

	def _absolute_path(self, path):
		return os.path.join(os.path.dirname(__file__), path) # 获得日志配置文件的绝对路径函数

	def _load(self):
		with open(self._config_path, 'r') as f:	# 载入日志配置
			self._conf = yaml.load(f) # 载入log文件保存地址等信息

	@property
	def conf(self):
		return self._conf

	def set_logging(self, for_web=True):
		"""配置logging"""
		log_dir = self._absolute_path('../logs/')
		lake.dir.mk(log_dir)
		log_config = self.conf['logging']
		update_filename(log_config) # 更新logging中filename的配置
		logging.config.dictConfig(log_config)


def update_filename(log_config):
	"""更新logging中filename的配置"""
	to_log_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), '../', x))
	if 'filename' in log_config:
		log_config['filename'] = to_log_path(log_config['filename'])
	for key, value in log_config.items():
		if isinstance(value, dict):
			update_filename(value)


config = ConfigLoader()








