# -*- coding: utf-8 -*-
"""
Created on 2020/3/12 10:27

@Project -> File: algorithm-tools -> symbolic_operations.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 函数符号计算
"""

from sympy import solve as symsolve
from sympy import symbols, diff

__doc__ = """
	# 算例:
	# 定义曲面函数.
	def S(x: list):
		y = x[1] ** 2 - x[0] ** 2
		return y
	
	S_dim = 2
	
	# 构造曲面函数S及其对应的显函数G.
	x_syms = gen_x_syms(S_dim)
	S_sym = gen_sym_S(S, x_syms)
	G_syms_list = gen_sym_G(S_sym, x_syms)
	
	# 计算偏导数符号.
	PD_syms = cal_partial_derive_syms(S_sym, S_dim)
	
	# 提取变量符号名.
	x_sym_strs = extract_x_syms_in_func_sym(S_sym, S_dim)
"""


def gen_x_syms(S_dim: int) -> list:
    """
    根据曲面维度生成所有变量符号x
    """
    x_syms = []
    for i in range(S_dim):
        x_syms.append(symbols('x_{}'.format(i)))
    return x_syms


def gen_sym_S(S, x_syms: list):
    """
    生成隐函数S的符号式
    * S(x_0, x_1, ..., x_n) = 0
    """
    S_sym = S(x_syms)
    return S_sym


def gen_sym_G(S_sym, x_syms: list) -> list:
    """
    求取隐函数S对应的显函数G符号式列表
    * x_n = G(x_0, x_1, ..., x_n-1)
    """
    G_syms_list = symsolve(S_sym, x_syms[-1])
    return G_syms_list


def cal_sym_func_value(F_sym, subs_dict: dict) -> float:
    """
    计算符号函数f的值
    :param f: sym func, 以symbol形式记录的函数
    :param subs_dict: dict, 符号函数中各变量和对应值字典, 例如{'x_0': 0.0, 'x_1': 1.0, ...}
    """
    v = float(F_sym.subs(subs_dict))
    return v


def cal_partial_derive_syms(F_sym, F_dim: int) -> list:
    """
    计算函数F的偏导数向量符号式
    """
    x_syms = gen_x_syms(F_dim)
    PD_syms = []
    for i in range(F_dim):
        PD_syms.append(diff(F_sym, x_syms[i]))
    return PD_syms


def extract_x_syms_in_func_sym(F_sym, F_dim: int) -> list:
    """
    提取函数符号F_sym中出现的所有自变量x的符号
    """
    x_sym_strs = []
    for i in range(F_dim):
        if symbols('x_{}'.format(i)) in F_sym.free_symbols:
            x_sym_strs.append('x_{}'.format(i))
    return x_sym_strs
