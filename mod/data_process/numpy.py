# -*- coding: utf-8 -*-
"""
Created on 2021/02/27 17:11:20

@File -> numpy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: numpy数据处理
"""

__all__ = [
    'min_max_normalize', 'min_max_normalize2', 'standard_normalize',
    'normalize_by_col',
    'convert_series_values',
    'add_noise', 'drop_nans_and_add_noise',
    'time_delayed_series',
    'savitzky_golay',
    'random_sampling',
]

from collections import defaultdict
from math import factorial
from typing import List
import pandas as pd
import numpy as np

import copy

EPS = 1e-6


# ---- 标准化 ---------------------------------------------------------------------------------------


def min_max_normalize(arr: np.ndarray):
    """
    对np.array每列按照min-max归一化

    :param arr: 待归一化的数据(表)
    """
    try:
        if arr.shape[1] == 1:
            v_min, v_max = np.min(arr), np.max(arr)
            arr = (arr - v_min) / (v_max - v_min)
            return arr, (v_min, v_max)
        elif arr.shape[1] > 1:
            for i in range(arr.shape[1]):
                v_min, v_max = np.min(arr[:, i]), np.max(arr[:, i])
                arr[:, i] = (arr[:, i] - v_min) / (v_max - v_min)
            return arr
        else:
            raise RuntimeError('归一化失败')
    except Exception as e:
        raise RuntimeError(e)


def min_max_normalize2(arr: np.ndarray, eps: float = 1e-6):
    """
    对np.array每列按照min-max归一化

    :param arr: 待归一化的数据(表)
    :param eps: 防止除数为0
    """
    try:
        if arr.shape[1] == 1:
            v_min, v_max = np.min(arr), np.max(arr)
            arr = (arr - v_min) / (v_max - v_min + eps)
            return arr, (v_min, v_max)
        elif arr.shape[1] > 1:
            v_min, v_max = np.min(arr, axis = 0), np.max(arr, axis = 0)
            arr = (arr - v_min) / (v_max - v_min + eps)
            return arr, (v_min, v_max)
        else:
            raise RuntimeError('归一化失败')
    except Exception as e:
        raise RuntimeError(e)


def standard_normalize(arr: np.ndarray, eps: float = 1e-6):
    """
    对np.array标准化处理

    :param arr: np.array, 待归一化的数据
    :param eps: 防止除数为0
    """
    try:
        if arr.shape[1] == 1:
            mean, std = np.mean(arr), np.std(arr)
            arr = (arr - mean) / (std + eps)
            return arr, (mean, std)
        elif arr.shape[1] > 1:
            mean, std = np.mean(arr), np.std(arr)
            arr = (arr - mean) / (std + eps)
            return arr, (mean, std)
        else:
            raise RuntimeError('归一化失败')
    except Exception as e:
        raise RuntimeError(e)


# ---- 数组归一化 -----------------------------------------------------------------------------------

def normalize_by_col(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    D = arr.shape[1]
    for i in range(D):
        arr[:, i] = min_max_normalize(arr[:, i])
    return arr


# ---- 值转换 ---------------------------------------------------------------------------------------

def _convert_series_label2number(x: list or np.ndarray):
    """将离散值序列中的类别转为数值"""
    x = np.array(x).flatten()
    _map = defaultdict(int)
    _labels = list(np.unique(x))
    for i in range(len(_labels)):
        _label = _labels[i]
        _map[_label] = i
    
    for k, v in _map.items():
        x[x == k] = int(v)
    
    x = np.array(x, dtype = np.float16)
    return x


def convert_series_values(x: list or np.ndarray, x_type: str):
    """单序列值转换, 连续值转为np.float64, 离散值转换为np.float16"""
    d_types = {'numeric': np.float64, 'categoric': np.float16}
    try:
        x = np.array(x, dtype = d_types[x_type]).flatten()
        return x
    except:
        if x_type == 'categoric':
            try:
                x = _convert_series_label2number(x)
                return x
            except:
                raise RuntimeError(
                    'Cannot convert x into numpy.ndarray with numerical values np.float64 or '
                    'np.float16'
                )
        else:
            raise RuntimeError(
                'Cannot convert x into numpy.ndarray with numerical values np.float64 or np.float16'
            )


# ---- 值处理 ---------------------------------------------------------------------------------------

def _add_noise(x: np.ndarray, x_type: str, noise_coeff: float, for_te: bool = False) -> (np.ndarray, np.ndarray):
    """
    生成数据样本
    :param x: 自变量序列
    :param x_type: x的数据类型
    :param noise_coeff: 噪音相对于标准差的系数, >= 0.0
    """
    x_cp = x.copy()

    if x_type == 'numeric':
        N, D = x_cp.shape
        std_x = np.std(x_cp, axis = 0).reshape(1, -1)
        noise_x = 2 * (np.random.random([N, D]) - 0.5)  # 介于-1到1的随机数组
        x_cp += noise_coeff * np.multiply(np.dot(np.ones([N, 1]), std_x), noise_x)

        if for_te:
            x_cp += noise_coeff * np.random.random([len(x_cp), 1])

    return x_cp


def add_noise(arr: np.ndarray, types: List[str], noise_coeff: float, for_te: bool = False):
    """
    逐列加入噪音
    :param arr: 二维数组, shape = (N, D)
    :param types:
    :param noise_coeff:
    :param for_te:
    :return:
    """
    arr = arr.astype(np.float32)
    x_arr = None
    for i in range(arr.shape[1]):
        x_ = _add_noise(arr[:, i].reshape(-1, 1), types[i], noise_coeff, for_te)
        if i == 0:
            x_arr = x_
        else:
            x_arr = np.hstack((x_arr, x_))
    return x_arr


def drop_nans_and_add_noise(arr: np.ndarray, var_types: list):
    """在数组中去掉异常值, 并对连续值加入噪声"""
    D = arr.shape[1]
    df = pd.DataFrame(arr).dropna(axis = 0, how = 'any')
    arr = np.array(df)
    
    for i in range(D):
        if var_types[i] == 'numeric':
            arr[:, i] += EPS * np.random.random(arr.shape[0])
    return arr


# ---- 时间延迟序列 ---------------------------------------------------------------------------------

def time_delayed_series(arr: np.ndarray, td_lag: int):
    """
    生成时间延迟序列

    :param arr: 样本数组, shape = (N, D = 2), 第一列为x, 第二列为y
    :param td_lag: 时间平移样本点数, 若td_lag > 0, 则x对应右方td_lag个样本点后的y;
                               若td_lag < 0, 则y对应右方td_lag个样本点后的x
    """
    lag_remain = np.abs(td_lag) % arr.shape[0]  # 整除后的余数
    x_td = copy.deepcopy(arr[:, 0])
    y_td = copy.deepcopy(arr[:, 1])
    
    if lag_remain == 0:
        pass
    else:
        if td_lag > 0:
            y_td = y_td[lag_remain:]
            x_td = x_td[:-lag_remain]
        else:
            x_td = x_td[lag_remain:]
            y_td = y_td[:-lag_remain]
    
        # if td_lag> 0:
        #     y_td = np.hstack((y_td[lag_remain:], y_td[:lag_remain]))
        # else:
        #     x_td = np.hstack((x_td[lag_remain:], x_td[:lag_remain]))
    
    return x_td, y_td


# ---- 滤波 ----------------------------------------------------------------------------------------

def savitzky_golay(y, window_size, order, deriv = 0, rate = 1) -> np.ndarray:
    """
    savitzky_golay滤波, ref: https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    """
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except Exception:
        raise ValueError("window_size and order have to be of type int")
    
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    
    # 预计算系数.
    b = np.mat([[k ** i for i in order_range]
                for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    
    # pad the signal at the extremes with values taken from the signal itself.
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    
    return np.convolve(m[::-1], y, mode = 'valid')


# ---- 采样 -----------------------------------------------------------------------------------------

def random_sampling(arr: np.ndarray, keep_n: int, seed: int = 0):
    arr = arr.copy()
    
    # 获取数据参数.
    N = arr.shape[0]
    idxes = list(range(N))
    
    # 随机采样.
    np.random.seed(seed)
    idxes = np.random.permutation(idxes)
    arr = arr[idxes[: keep_n]][:]
    
    return arr
