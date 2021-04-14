# -*- coding: utf-8 -*-
"""
Created on 2020/1/21 下午2:27

@Project -> File: pollution-forecast-offline-training-version-2 -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 初始化项目
"""

import yaml
import os

proj_dir = os.path.abspath(os.path.dirname(__file__))

with open('proj_dir_struc.yml', 'r', encoding = 'utf-8') as f:
	fp_dict = yaml.load(f, Loader = yaml.Loader)


def mk_dir(root, fp_dict):
	"""使用递归算法创建文件目录"""
	if fp_dict is None:
		pass
	elif type(fp_dict) == dict:
		for key in fp_dict.keys():
			sub_fp = os.path.join(root, key)
			sub_fp_dict = fp_dict[key]
			
			# 如果要创建的目录不存在则新建.
			if key not in os.listdir(root):
				os.mkdir(sub_fp)
			sub_root = sub_fp
			mk_dir(sub_root, sub_fp_dict)
	else:
		raise RuntimeError('Invalid file path dict type: {}.'.format(type(fp_dict)))


def mk_file(file_dir, file_name):
	if file_name not in os.listdir(file_dir):
		with open(os.path.join(file_dir, file_name), 'w') as f:
			f.write('')


# 初始化项目目录结构.
mk_dir(proj_dir, fp_dict['root'])

# 初始化.gitignore文件.
mk_file(proj_dir, '.gitignore')

# 初始化config目录文件.
config_dir = os.path.join(proj_dir, 'config/')
for file in ['model_config.yml', 'default.yml', 'test_params.yml']:
	mk_file(config_dir, file)

	

	




