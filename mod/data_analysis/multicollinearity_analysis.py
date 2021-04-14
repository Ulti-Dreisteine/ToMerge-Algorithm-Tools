# -*- coding: utf-8 -*-
"""
Created on 2020/1/22 下午4:02

@Project -> File: algorithm-tools -> multicollinearity_analysis.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 多重共线性分析
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from lake.decorator import time_cost
import pandas as pd
import numpy as np
import copy


def cal_vif(y_true, y_pred):
    """计算方差膨胀因子， VIF = 1 / (1 - R^2)."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    r2 = r2_score(y_true, y_pred)

    if r2 == 1:
        return 9999
    else:
        vif = 1 / (1 - r2)
        return vif


@time_cost
def multicollinearity_test(data, cols, vif_thres=10.0, cut_off=None):
    """
    当data表中的字段间可能存在强相关性时, 通过方差膨胀因子VIF进行特征筛选.
    :param data: pd.DataFrame, 待分析数据表
    :param vif_thres: float > 0.0, default 10.0, VIF阈值
    :param cut_off: int, 截断因子, 认为超过cut_off值的数据间没有相关性, 然后进行截断, 可以减少线性回归计算量
    :return: samples_arr: np.array, 选中字段构成的样本array
    :return: chosen_cols: list of strs, 选中字段名构成的list

    Note:
            1. 优化计算速度，考虑到前后相距过长的变量间不会具有相关性, 引入cut_off进行截断操作

    Example:
    ------------------------------------------------------------
    import sys
    import os

    sys.path.append('../..')

    from lib import proj_dir

    data = pd.read_csv(os.path.join(proj_dir, 'data/provided/weather/data_denoised.csv'))

    # %% 进行共线性分析
    cols = data.columns
    samples_arr, chosen_cols = multicollinearity_test(data, cols, cut_off = 50)
    ------------------------------------------------------------
    """

    assert type(data) == pd.DataFrame
    data = data.copy()

    chosen_cols = [cols[0]]
    chosen_samples_arr = data.loc[:, chosen_cols[0]].to_numpy().reshape(-1, 1)
    print('choose column {}'.format(cols[0]))

    for col in cols[1:]:
        if cut_off:
            # 只取时间上相近的数据做分析.
            if chosen_samples_arr.shape[1] > cut_off:
                samples_arr = copy.deepcopy(chosen_samples_arr[:, -cut_off:])
            else:
                samples_arr = copy.deepcopy(chosen_samples_arr)
        else:
            samples_arr = copy.deepcopy(chosen_samples_arr)

        # 候选特征.
        candidate_arr = data.loc[:, col].to_numpy().reshape(-1, 1)

        # 建立线性回归模型，计算VIF值.
        rgsr = LinearRegression()
        rgsr.fit(samples_arr, candidate_arr)

        y_pred = rgsr.predict(samples_arr)
        vif = cal_vif(candidate_arr, y_pred)

        # 根据VIF结果判断共线性程度.
        if vif < vif_thres:
            print('choose column {}'.format(col))
            chosen_samples_arr = np.hstack((chosen_samples_arr, candidate_arr))
            chosen_cols.append(col)
        else:
            continue

    return chosen_samples_arr, chosen_cols
