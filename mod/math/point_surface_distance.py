# -*- coding: utf-8 -*-
"""
Created on 2020/3/12 10:03

@Project -> File: algorithm-tools -> point_surface_distance.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 计算点到任意维度曲面的距离
"""

from scipy.optimize import fsolve
import numpy as np

from .partial_derives import PartialDerives

__doc__ = """
	# 算例：
	# 定义函数和参数.
	def func(x: list):
		y = x[1] - x[0] ** 3
		return y
	
	# 测试.
	f_dim = 2
	xps = [0.5, 2]
	x0 = [1.5]
	x_opt, dist = cal_distance2surface(func, f_dim, xps, x0)
	
	# 画图验证.
	import matplotlib.pyplot as plt
	
	x = np.arange(-3, 3 + 0.1, 0.1)
	y = np.power(x, 3)
	
	plt.figure(figsize = [8, 8])
	plt.plot(x, y, label = 'surface')
	plt.scatter(xps[0], xps[1], s = 12, label = 'point')
	plt.scatter(x_opt[0], x_opt[1], s = 12, c = 'black', label = 'closest point on the surface')
	plt.legend(loc = 'upper left')
	plt.xlim([-3, 3])
	plt.ylim([-3, 3])
	plt.xlabel('x')
	plt.ylabel('y')
	plt.grid(True)
"""

EPS = 1e-6


def cal_distance2surface(func, f_dim: int, xps: list, x0: list) -> np.array:
	"""计算高维空间中点xps离函数func(x) = 0构成的曲面的距离和最近点坐标"""
	partial_derives = PartialDerives(func, f_dim)
	
	def _eqs2solve(x):
		x = list(x).copy()
		x, pd_values = partial_derives.cal_partial_derive_values(x)
		eqs = []
		for i in range(f_dim - 1):
			if pd_values[-1] == 0:
				pd_values[-1] = EPS
			if pd_values[i] == 0:
				pd_values[i] = EPS
			e_ = (xps[i] - x[i]) / (pd_values[i] / pd_values[-1]) - xps[-1] + x[-1]
			eqs.append(e_)
		return eqs
	
	root = fsolve(_eqs2solve, np.array(x0))
	
	# 计算曲线上最近邻点.
	x_opt, _ = partial_derives.cal_partial_derive_values(list(root))
	dist = np.linalg.norm(x_opt - np.array(xps))
	
	return x_opt, dist
