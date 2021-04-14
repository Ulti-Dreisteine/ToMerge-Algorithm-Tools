# -*- coding: utf-8 -*-
"""
Created on 2020/3/12 11:16

@Project -> File: algorithm-tools -> partial_derives.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 计算函数偏导数
"""

from .symbolic_operations import gen_x_syms, gen_sym_S, gen_sym_G, cal_partial_derive_syms, \
	extract_x_syms_in_func_sym, cal_sym_func_value


class PartialDerives(object):
	"""
	计算隐函数各维度偏导数的表达式和值
	
	example:
	---
	def f(x: list):
		y = 0.5 * x[1] - x[0] ** 2
		return y
	
	f_dim = 2
	partial_derivs = PartialDerives(f, f_dim)
	x, pd_values = partial_derivs.cal_partial_derive_values([1])
	"""
	
	def __init__(self, func, f_dim):
		"""
		初始化
		:param func: func, 函数必须为转换为隐函数形式输入, func(x) = 0
		:param f_dim: int, 函数里自变量的维数
		"""
		self.F = func
		self.F_dim = f_dim
		
		# 生成变量和函数的偏导数符号.
		self.x_syms = gen_x_syms(self.F_dim)
		self.F_sym = gen_sym_S(self.F, self.x_syms)
		self.G_syms_list = gen_sym_G(self.F_sym, self.x_syms)
		self.G_sym = self.G_syms_list[0]  # TODO: 目前默认选择第一个解, 需要改为对所有解均进行计算
	
	def _cal_partial_derive_syms(self) -> list:
		"""
		计算隐函数F对各x的偏导符号
		"""
		PD_syms = cal_partial_derive_syms(self.F_sym, self.F_dim)
		return PD_syms
	
	@property
	def PD_syms(self):
		return self._cal_partial_derive_syms()
	
	def _extract_x_syms_in_PD(self) -> list:
		"""
		提取PD_syms中各个符号表达式中所含自变量x符号
		"""
		PD_have_x_syms = []
		for i in range(self.F_dim):
			PD_ = self.PD_syms[i]
			x_sym_strs_ = extract_x_syms_in_func_sym(PD_, self.F_dim)
			PD_have_x_syms.append(x_sym_strs_)
		return PD_have_x_syms
	
	def _extract_x_syms_in_G(self) -> list:
		G_has_x_syms = extract_x_syms_in_func_sym(self.G_sym, self.F_dim)
		return G_has_x_syms
	
	@property
	def PD_have_x_syms(self):
		return self._extract_x_syms_in_PD()
	
	@property
	def G_has_x_syms(self):
		return self._extract_x_syms_in_G()
	
	def cal_partial_derive_values(self, x: list) -> (list, list):
		"""
		计算偏导数
		* x不需要输入最后x_n的值, 可以通过G函数计算, 所以 x = [x_0, x_1, ..., x_n-1]
		"""
		x = x.copy()
		assert len(x) == self.F_dim - 1
		
		# 计算G的值.
		subs_dict_ = {}
		for x_sym_str in self.G_has_x_syms:
			subs_dict_[x_sym_str] = x[int(x_sym_str.split('_')[1])]
		x_end = cal_sym_func_value(self.G_sym, subs_dict_)
		x.append(x_end)
		
		pd_values = []
		for i in range(self.F_dim):
			subs_dict_ = {}
			for x_sym_str in self.PD_have_x_syms[i]:
				subs_dict_[x_sym_str] = x[int(x_sym_str.split('_')[1])]
			pd_ = float(self.PD_syms[i].subs(subs_dict_))
			pd_values.append(pd_)
		
		return x, pd_values
