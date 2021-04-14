# -*- coding: utf-8 -*-
"""
Created on 2020/7/9 12:32 下午

@File: univariate.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 单变量分箱
"""
from lake.decorator import time_cost
import pandas as pd
import numpy as np
import logging
import warnings
from typing import Union

from ..data_binning import VALUE_TYPES_AVAILABLE, METHODS_AVAILABLE, convert_series_values

logging.basicConfig(level=logging.INFO)


def _cal_stat_characters(x: list or np.ndarray, x_type: str):
    """计算连续值变量统计学特征"""
    x = np.array(x).flatten()
    try:
        assert x_type == 'continuous'
    except:
        warnings.warn(
            'WARNING: stat characters may not be accurate for self.x_type = {}'.format(x_type))

    _mean = np.mean(x)
    _std = np.std(x)
    _q1, _q2, _q3 = np.percentile(x, (25, 50, 75), interpolation='midpoint')
    _iqr = abs(_q3 - _q1)

    stat_params = {
        'mean': _mean,
        'std': _std,
        'percentiles': {
            'q1': _q1,  # 下25%位
            'q2': _q2,  # 中位数
            'q3': _q3,  # 上25%位
            'iqr': _iqr
        }
    }
    return stat_params


class UnivarBinning(object):
    """
    单变量分箱

    Example:
    ------------------------------------------------------------------------------------------------
    univar_bin = UnivarBinning(x, x_type)
    test_results = defaultdict(dict)
    test_params = {
            'label': {},
    }

    freq_ns, labels = univar_bin.label_binning(**test_params['label'])
    """

    def __init__(self, x: list or np.ndarray, x_type: str in VALUE_TYPES_AVAILABLE):
        """
        :param x: 待分箱序列
        :param x_type: 序列值类型, 必须在VALUE_TYPES_AVAILABLE中选择
        """
        if x_type not in VALUE_TYPES_AVAILABLE:
            raise ValueError('Invalid x_type {}'.format(x_type))

        # 将序列值转换为数值， 连续值转为np.float64, 离散值转换为np.float16.
        self.x = convert_series_values(x, x_type)

        # 默认删除了数据中的nan值
        self.x = self.x[~np.isnan(self.x)]
        self.x_type = x_type

    @property
    def stat_characters(self) -> dict:
        """单变量数据统计学特征"""
        return _cal_stat_characters(self.x, self.x_type)

    @property
    def binning_bounds(self) -> list:
        _percentiles = self.stat_characters['percentiles']
        _q3, _q1, _iqr = _percentiles['q3'], _percentiles['q1'], _percentiles['iqr']

        # 使用分位数确定序列分箱数据值范围.
        # TODO: 这里还不能使用分位数.
        # bounds = [
        # 	max(np.min(self.x), _q1 - 1.5 * _iqr),
        # 	min(np.max(self.x), _q3 + 1.5 * _iqr)
        # ]
        bounds = [np.min(self.x), np.max(self.x)]
        return bounds

    def _check_binning_match(self, current_method: str, suit_x_type: str, suit_method: str):
        """检查分箱方法与待分箱值类型是否匹配"""
        try:
            assert self.x_type == suit_x_type
        except Exception:
            warnings.warn(
                'x_type is not "{}" for self.{}, try switch to self.{}'.format(
                    self.x_type, current_method, suit_method)
            )

    # @time_cost
    def isometric_binning(self, bins: int) -> Union[list, list]:
        """
        连续数据等距分箱
        :param bins: 分箱个数
        """
        self._check_binning_match(
            'isometric_binning', 'continuous', 'label_binning')

        # 分箱, 数据只会在分箱边界内进行分箱.
        freq_ns, _intervals = np.histogram(
            self.x, bins, range=self.binning_bounds)
        labels = _intervals[1:]  # **以每个分箱区间的右边界为label

        # 转为list类型.
        # TODO: 此处的数值精度是否能够足够用于区分, 需确认是否需要将数据进行归一化处理.
        freq_ns = list(freq_ns)
        labels = list(labels.astype(np.float32))

        return freq_ns, labels

    # @time_cost
    def equifreq_binning(self, equi_freq_n: int) -> Union[list, list]:
        """
        等频分箱
        :param equi_freq_n: 分箱箱子的样本数(上限)
        """
        self._check_binning_match(
            'equifreq_binning', 'continuous', 'label_binning')
        x = list(np.sort(self.x))

        freq_ns, labels = [], []
        while True:
            if len(x) <= equi_freq_n:               # 将该箱与上一个箱合并
                freq_ns[-1] += len(x)
                labels[-1] = x[-1]
                break
            else:
                freq_ns.append(equi_freq_n)
                labels.append(x[equi_freq_n - 1])
                x = x[equi_freq_n:]
                continue

        return freq_ns, labels

    # @time_cost
    def quasi_chi2_binning(self, init_bins: int, final_bins: int, merge_freq_thres: float = None) -> Union[list, list]:
        """
        连续数据拟卡方分箱
        :param init_bins: 初始分箱数
        :param final_bins: 最终分箱数下限
        :param merge_freq_thres: 合并分箱的密度判据阈值
        """
        self._check_binning_match(
            'quasi_chi2_binning', 'continuous', 'label_binning')

        if merge_freq_thres is None:
            merge_freq_thres = len(self.x) / init_bins / 10  # 默认分箱密度阈值

        # 初始化.
        init_freq_ns, init_labels = self.isometric_binning(init_bins)
        densities = init_freq_ns                            # 这里使用箱频率密度表示概率分布意义上的密度
        init_box_lens = [1] * init_bins

        # 根据相邻箱密度差异判断是否合并箱.
        bins = init_bins
        freq_ns = init_freq_ns
        labels = init_labels
        box_lens = init_box_lens

        while True:
            do_merge = 0

            # 在一次循环中优先合并具有最高相似度的箱.
            similar_ = {}
            for i in range(bins - 1):
                j = i + 1
                density_i, density_j = densities[i], densities[j]
                s = abs(density_i - density_j)              # 密度相似度，

                if s <= merge_freq_thres:
                    similar_[i] = s
                    do_merge = 1
                else:
                    continue

            if (do_merge == 0) | (bins == final_bins):
                break
            else:
                similar_ = sorted(similar_.items(),
                                  key=lambda x: x[1], reverse=False)  # 升序排列
                i = list(similar_[0])[0]
                j = i + 1

                # 执行i和j箱合并, j合并到i箱
                freq_ns[i] += freq_ns[j]
                box_lens[i] += box_lens[j]
                densities[i] = freq_ns[i] / box_lens[i]     # 使用i、j箱混合后的密度
                labels[i] = labels[j]

                freq_ns = freq_ns[: j] + freq_ns[j + 1:]
                densities = densities[: j] + densities[j + 1:]
                labels = labels[: j] + labels[j + 1:]
                box_lens = box_lens[: j] + box_lens[j + 1:]

                bins -= 1

        return freq_ns, labels

    # @time_cost
    def label_binning(self) -> Union[list, list]:
        """根据离散数据自身标签值进行分箱"""
        self._check_binning_match(
            'label_binning', 'discrete', 'isometric_binning/quasi_chi2_binning/equifreq_binning')

        labels = sorted(list(set(self.x)))  # **按照值从小到大排序

        # 统计freq_ns.
        _df = pd.DataFrame(self.x, columns=['label'])
        _df['index'] = _df.index
        _freq_counts = _df.groupby('label').count()
        _freq_counts = _freq_counts.to_dict()['index']

        freq_ns = []
        for i in range(len(labels)):
            freq_ns.append(_freq_counts[labels[i]])

        return freq_ns, labels

    def univar_binning(self, method: str, **params):
        """序列分箱算法"""
        freq_ns, labels = None, None
        if method in METHODS_AVAILABLE['continuous'] + METHODS_AVAILABLE['discrete']:
            if method == 'isometric':
                freq_ns, labels = self.isometric_binning(params['bins'])
            elif method == 'equifreq':
                freq_ns, labels = self.equifreq_binning(params['equi_freq_n'])
            elif method == 'quasi_chi2':
                freq_ns, labels = self.quasi_chi2_binning(
                    params['init_bins'], params['final_bins'])
            elif method == 'label':
                freq_ns, labels = self.label_binning()
            return freq_ns, labels
        else:
            raise ValueError('Invalid method {}'.format(method))

# TODO: 在test中编写对应的测试代码.

# if __name__ == '__main__':
#     # ============ 载入测试数据和参数 ============
#     from collections import defaultdict
#     from lib import load_test_data

#     data = load_test_data(label='patient')

#     # ============ 测试连续值分箱 ============
#     col = 'CK'
#     x_type = 'continuous'
#     x = np.array(data[col])

#     self = UnivarBinning(x, x_type)

#     test_results = defaultdict(dict)
#     test_params = {
#         'isometric': {'bins': 30},
#         'equifreq': {'equi_freq_n': 20},
#         'quasi_chi2': {'init_bins': 150, 'final_bins': 30}
#     }

#     # 测试各分箱函数.
#     for method in ['isometric', 'equifreq', 'quasi_chi2']:
#         if method == 'isometric':
#             freq_ns, labels = self.isometric_binning(**test_params[method])
#         elif method == 'equifreq':
#             freq_ns, labels = self.equifreq_binning(**test_params[method])
#         elif method == 'quasi_chi2':
#             freq_ns, labels = self.quasi_chi2_binning(**test_params[method])
#         test_results[method] = {'freq_ns': freq_ns, 'labels': labels}

#     # 测试通用分箱函数.
#     for method in ['isometric', 'equifreq', 'quasi_chi2']:
#         freq_ns, labels = self.univar_binning(method, **test_params[method])
#         test_results['by_general_func'][method] = {
#             'freq_ns': freq_ns, 'labels': labels}

#     # ============ 测试离散值分箱 ============
#     # col = 'SEX'
#     # x_type = 'discrete'
#     # x = np.array(data[col])
#     #
#     # self = UnivarBinning(x, x_type)
#     #
#     # test_results = defaultdict(dict)
#     # test_params = {
#     # 	'label': {},
#     # }
#     #
#     # freq_ns, labels = self.label_binning(**test_params['label'])
