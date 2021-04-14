# -*- coding: utf-8 -*-
'''
Created on 2021/02/27 16:30:25

@File -> test_ga_optimization.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 测试遗传算法代码
'''

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../'))
sys.path.append(BASE_DIR)

from mod.numerical_optimization import cal_distance2surface, GeneticAlgorithm


# ---- 目标函数 -------------------------------------------------------------------------------------

def obj_func(x):
    # return np.linalg.norm(x, 2)
    return x[1] - 0.5 * x[0]

# ---- 约束条件 -------------------------------------------------------------------------------------

# 约束函数边界方程.


def c0(x):
    return x[1] + 0 * x[0] - 0.5


def c1(x):
    return x[1] - x[0] - 0.75

# 约束条件函数.


def constr_0(x):
    if c0(x) >= 0:
        return 0
    else:
        f_dim = 2
        _, d = cal_distance2surface(c0, f_dim, x, x[:-1])
        return 1e6 + 1e12 * d


def constr_1(x):
    if c1(x) >= 0:
        return 0
    else:
        f_dim = 2
        _, d = cal_distance2surface(c1, f_dim, x, x[:-1])
        return 1e6 + 1e12 * d


# ---- 遗传算法优化 ---------------------------------------------------------------------------------

# 设定参数.
chrome_len = 2
chrome_bounds = [[-1, 1], [-1, 1]]
chrome_types = [1, 1]
pop_size = 300
pc = 0.4
pm = 0.2
optim_direc = 'minimize'
epochs = 1000

# 进行优化.
self = GeneticAlgorithm(chrome_len, chrome_bounds,
                        chrome_types, pop_size, pc, pm)
final_fitness, x_opt, eval_process = self.evolution(
    lambda x: obj_func(x) + constr_0(x) + constr_1(x),
    optim_direc=optim_direc,
    epochs=epochs,
    max_no_change=20
)

# 优化结果效果显示.
import matplotlib.pyplot as plt
h = 0.01
x = np.arange(-1, 1 + h, h)
y_c0 = -0 * x + 0.5  # 约束1的边界
y_c1 = x + 0.75  # 约束2的边界

plt.scatter(x_opt[0], x_opt[1], s=20)
plt.plot(x, y_c0, 'k--', linewidth=0.3)
plt.plot(x, y_c1, 'k--', linewidth=0.3)
# plt.xlim([0.35, 0.65])
# plt.ylim([0.6, 1])
