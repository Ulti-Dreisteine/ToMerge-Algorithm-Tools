# -*- coding: utf-8 -*-
"""
Created on 2020/7/9 12:58 下午

@File: multivar_binning.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 多元变量联合分箱
"""

from typing import Union
import numpy as np
import logging
import warnings

from .univar_binning import UnivarBinning
from ..data_binning import VALUE_TYPES_AVAILABLE, METHODS_AVAILABLE, EPS, convert_series_values, drop_nans_and_add_noise

logging.basicConfig(level=logging.INFO)


def _check_var_type_and_method(var_type, method):
    if var_type == 'continuous':
        assert method in METHODS_AVAILABLE[var_type]
    elif var_type == 'discrete':
        assert method in METHODS_AVAILABLE[var_type]
    else:
        raise ValueError('ERROR: unknown value_type {}'.format(var_type))


def _check_array_and_params(arr, var_types, methods, params):
    # 检查数据格式.
    try:
        N, D = arr.shape
        assert (N > 0) & (D > 1)
    except:
        raise ValueError('ERROR: data_shape = {}'.format(arr.shape))

    # 检查值类型.
    try:
        for t in var_types:
            assert t in VALUE_TYPES_AVAILABLE
    except:
        raise ValueError('ERROR: var_types = {}'.format(var_types))

    # 检查分箱方法.
    try:
        all_methods = []
        for key in METHODS_AVAILABLE.keys():
            m = METHODS_AVAILABLE[key]
            if type(m) == str:
                all_methods.append(m)
            elif type(m) == list:
                all_methods += m
            else:
                raise ValueError(
                    'ERROR: invalid type(m) == {}'.format(type(m)))

        for method in methods:
            assert method in all_methods
    except:
        raise ValueError('ERROR: methods = {}'.format(methods))

    # 检查数据维数.
    try:
        assert len(var_types) == arr.shape[1]
        assert len(methods) == arr.shape[1]
        assert len(params) == arr.shape[1]
    except:
        raise ValueError('Array dims = {}, var_types length = {}, methods length = {}'.format(
            arr.shape[1], len(var_types), len(methods)))

    # 检查value_type与method匹配关系.
    try:
        for i in range(arr.shape[1]):
            value_type = var_types[i]
            method = methods[i]
            _check_var_type_and_method(value_type, method)
    except:
        warnings.warn("WARNING: var_types and methods don't match")


class MultivarBinning(object):
    """多元变量联合分箱"""

    def __init__(self, arr: np.ndarray, var_types: list):
        """
        :param arr: shape = (N, D), N为数据样本数, D为样本维数
        :param var_types: 各维数上的数据类型
        """
        self.D = arr.shape[1]

        arr_ = []
        for i in range(self.D):
            arr_.append(convert_series_values(arr[:, i], var_types[i]))
        arr_ = np.array(arr_).T

        self.arr = drop_nans_and_add_noise(arr_, var_types)
        self.var_types = var_types
        self.N = self.arr.shape[0]

    def joint_binning(self, methods: list, params: list) -> Union[np.ndarray, list]:
        """
        联合分箱
        :param methods: 各维数采用分箱方式
        :param params: list of dicts, 各维数对应分箱计算参数
        :return:
                hist: 高维分箱后箱子中的频数
                edges: list of lists, 各维度上的标签
        """
        _check_array_and_params(self.arr, self.var_types, methods, params)

        # 各维度序列边际分箱.
        self.edges, self.binning_bounds = [], {}
        for d in range(self.D):
            binning_ = UnivarBinning(self.arr[:, d], x_type=self.var_types[d])
            _, e_ = binning_.univar_binning(method=methods[d], **params[d])
            self.edges.append(e_)

            if self.var_types[d] == 'continuous':
                self.binning_bounds[d] = binning_.binning_bounds

        # 在各个维度上将数据值向label进行插入, 返回插入位置.
        # 这里的arr值需要限制在序列插值范围内, 与UnivarBinning.binning_bounds范围对应.
        insert_locs_ = np.zeros_like(self.arr, dtype=int)

        for d in range(self.D):
            _series = self.arr[:, d]

            if self.var_types[d] == 'continuous':
                _series[_series < self.binning_bounds[d]
                        [0]] = self.binning_bounds[d][0]
                _series[_series > self.binning_bounds[d]
                        [1]] = self.binning_bounds[d][1]

            insert_locs_[:, d] = np.searchsorted(
                self.edges[d], _series, side='left')

        # 将高维坐标映射到一维坐标上, 然后统计各一维坐标上的频率.
        edges_len_ = list(np.max(insert_locs_, axis=0) + 1)
        ravel_locs_ = np.ravel_multi_index(insert_locs_.T, dims=edges_len_)
        hist = np.bincount(ravel_locs_, minlength=np.array(edges_len_).prod())

        # reshape转换形状.
        hist = hist.reshape(edges_len_)

        return hist, self.edges

# TODO: 在test中编写对应测试代码.
# if __name__ == '__main__':
# 	# ============ 载入测试数据和参数 ============
# 	from collections import defaultdict
# 	from lib import load_test_data

# 	data = load_test_data(label = 'patient')

# 	# ============ 准备参数 ============
# 	x_col = 'CK'
# 	y_col = 'VTE'
# 	x, y = list(data[x_col]), list(data[y_col])
# 	x_type, y_type = 'continuous', 'discrete'

# 	arr = np.vstack((x, y)).T
# 	var_types = [x_type, y_type]

# 	methods, params = [], []
# 	for i in range(2):
# 		if var_types[i] == 'continuous':
# 			methods.append('isometric')
# 			params.append({'bins': 10})
# 		else:
# 			methods.append('label')
# 			params.append({})

# 	# ============ 测试类 ============
# 	self = MultivarBinning(arr, var_types)
# 	hist, edges = self.joint_binning(methods, params)
