# -*- coding: utf-8 -*-
"""
Created on 2021/02/26 16:30

@Project -> File: general-algorithm-tools -> stat_tools.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 统计工具
"""

__all__ = ['numeric_stat_params', 'confusion_matrix', 'log_likelihood']

import numpy as np
import warnings

EPS = 1e-6


def numeric_stat_params(x: list or np.ndarray, x_type: str):
	"""计算连续值变量统计学特征"""
	x = np.array(x).flatten()
	try:
		assert x_type == 'numeric'
	except:
		warnings.warn(
			'WARNING: stat characters may not be accurate for self.x_type = {}'.format(x_type))

	mean = np.mean(x)
	std = np.std(x)
	q1, q2, q3 = np.percentile(x, (25, 50, 75), interpolation = 'midpoint')
	iqr = abs(q3 - q1)

	stat_params = {
		'mean': mean,
		'std': std,
		'percentiles': {
			'q1': q1,  # 下分位
			'q2': q2,  # 中位数
			'q3': q3,  # 上分位
			'iqr': iqr
		}
	}
	return stat_params


def confusion_matrix(two_dim_arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
	"""二维离散数组混淆矩阵计算"""
	c_matrix = np.zeros([rows, cols])
	for i in range(rows):
		for j in range(cols):
			c_matrix[i, j] = two_dim_arr[(two_dim_arr[:, 0] == i) & (two_dim_arr[:, 1] == j)].shape[0]
	return c_matrix


def log_likelihood(c_matrix: np.ndarray):
	"""基于混淆矩阵计算对数似然函数"""
	row_sums = np.sum(c_matrix, axis = 1).reshape(-1, 1)
	row_sums = np.tile(row_sums, [1, c_matrix.shape[1]])  # 在列方向复制多次展开
	row_probs = np.divide(c_matrix, row_sums + EPS)  # 点除
	log_like = np.sum(np.multiply(c_matrix, np.log(row_probs + EPS)))
	return log_like
