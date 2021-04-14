# -*- coding: utf-8 -*-
'''
Created on 2021/02/27 18:10:29

@File -> test_univar_encoding.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 测试单变量编码
'''

import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../'))
sys.path.append(BASE_DIR)

from src.settings import PROJ_DIR
from mod.data_encoding.univar_encoding import UnsuperCategorEncoding


def load_weather_samples(x_col: str, y_col: str) -> pd.DataFrame:
    continuous_cols = ['pm25', 'pm10', 'so2', 'co',
                       'no2', 'o3', 'aqi', 'ws', 'temp', 'sd']
    discrete_cols = ['weather', 'wd', 'month', 'weekday', 'clock_num']

    def _determ_dtype(col: str):
        if col in continuous_cols:
            return "continuous"
        elif col in discrete_cols:
            return "discrete"
        else:
            raise ValueError("Invalid col = '{}'".format(col))

    f_path = os.path.join(PROJ_DIR, 'data/weather/data.csv')
    data = pd.read_csv(f_path)

    x = data.loc[:, x_col].values.reshape(-1, 1)
    y = data.loc[:, y_col].values.reshape(-1, 1)

    x_type, y_type = _determ_dtype(x_col), _determ_dtype(y_col)

    samples_info = {
        "x": x,
        "y": y,
        "x_type": x_type,
        "y_type": y_type,
    }
    return samples_info


# ---- 测试 ----------------------------------------------------------------------------------------

x_col = "wd"
y_col = "pm25"
samples_info = load_weather_samples(x_col, y_col)
xs, ys = samples_info["x"], samples_info["y"]
x_type, y_type = samples_info["x_type"], samples_info["y_type"]

enc = UnsuperCategorEncoding(xs)
Xs = enc.encode(method="freq")
