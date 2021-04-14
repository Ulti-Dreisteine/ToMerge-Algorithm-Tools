# -*- coding: utf-8 -*-
"""
Created on 2019/12/24 下午12:27

@Project -> File: guodian-desulfuration-optimization -> genetic_algorithm.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 遗传算法
"""

import logging

logging.basicConfig(level=logging.INFO)

from typing import Tuple
from lake.decorator import time_cost
import numpy as np

from ..math.point_surface_distance import cal_distance2surface


def _exp_adj_func(x, w=10.0):
    """
    非线性调整项，将数值映射到0-1区间
    """
    x = (np.exp(w * x) - 1.0) / (np.exp(w) - 1)
    return x


class GeneticAlgorithm(object):
    """
    遗传算法, 用于求解非线性数值优化问题

    Note:
            1. 优化问题的输入X可以是连续值、离散类别或者二者混合, 输出y也可以是两种类型混合
            2. GA算法本身原理是不含约束的, 如果求解问题中含有约束, 可以从以下方式解决:
                    *	对自变量X约束, 可以在chrome_bounds中设定对应被约束变量的上下界边界值
                    *	对输出y约束, 可以对应约束条件利用y输出值构造惩罚函数, 惩罚函数中构造了
                            约束边界的梯度, 这样样本点会在迭代过程中沿着约束梯度方向划入可行域
    """

    def __init__(self, chrome_len, chrome_bounds, chrome_types, pop_size, pc=0.4, pm=0.2):
        """
        初始化
        :param chrome_bounds: list of lists, 染色体上各位置取值范围, 连续值对应上下界, 离散值对应所有可能取值情况
        :param chrome_types: list of ints from [0, 1], 用于标记染色体上各个元素数据类型，1为连续，0为离散
        :param chrome_len: int, 染色体长度，对应待优化参数维数
        :param pop_size: int, 种群数量
        :param pc: float in [0.0, 1.0], 交配概率，推荐为0.4
        :param pm: float in [0.0, 1.0], 突变概率, 推荐为0.2
        """
        if (len(chrome_bounds) != chrome_len) | (len(chrome_types) != chrome_len):
            raise ValueError('边界或类型设置与维数不一样长')

        self.chrome_len = chrome_len
        self.chrome_bounds = chrome_bounds
        self.chrome_types = chrome_types
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self._init_pop()

    def _gen_rand_values(self, i):
        """产生第i列的随机量"""
        rand_values = None
        col_bounds, col_type = self.chrome_bounds[i], self.chrome_types[i]

        if col_type == 1:
            rand_values = np.random.random(self.pop_size)
            rand_values = rand_values * \
                (col_bounds[1] - col_bounds[0]) + col_bounds[0]
        elif col_type == 0:
            rand_values = np.random.choice(col_bounds, self.pop_size)
        rand_values = rand_values.reshape(-1, 1)
        rand_values = np.array(rand_values, dtype=np.float32)

        return rand_values

    def _init_pop(self):
        """
        初始化种群
        :param chrome_bounds: refer to the definition of chrome_bounds in __init__(self)
        :param chrome_bounds: refer to the definition of chrome_bounds in __init__(self)
        """
        for i in range(self.chrome_len):
            col_bounds, col_type = self.chrome_bounds[i], self.chrome_types[i]
            if col_type == 1:  # 连续值
                if (len(col_bounds) != 2) | (col_bounds[0] >= col_bounds[1]):
                    raise ValueError('第{}个变量为连续数值，其边界参数不正确'.format(i))
            elif col_type == 0:  # 离散值
                pass
            else:
                raise ValueError('数值类型参数有误')

        self.pop = None
        for i in range(self.chrome_len):
            col_rand_values = self._gen_rand_values(i)
            if i == 0:
                self.pop = col_rand_values
            else:
                self.pop = np.hstack((self.pop, col_rand_values))

    def _cal_fitness(self, func, optim_direc, normalize=True):
        """
        计算适应度
        :param obj_func: function, 目标函数
        :param optim_direc: str from {'minimize', 'maximize'}, 优化方向
        :param normalize: bool, 是否进行0-1归一化处理
        :return: fitness, normalized_fitness(or fitness if not normalized)
        """
        fitness = np.apply_along_axis(func, 1, self.pop)

        if optim_direc == 'minimize':
            fitness = 1 / fitness
        elif optim_direc == 'maximize':
            pass
        else:
            raise ValueError('optim_direc参数不正确')

        if normalize:
            min_fit, max_fit = np.min(fitness), np.max(fitness)
            if min_fit == max_fit:
                normalized_fitness = np.ones_like(fitness)
            else:
                normalized_fitness = (
                    fitness.copy() - min_fit) / (max_fit - min_fit)
            return fitness, normalized_fitness
        else:
            return fitness, fitness

    @staticmethod
    def _get_accum_prob(fitness):
        r"""
        将适应度转化为累计概率
        :param fitness: np.array, 一维适应度表序列
        :return: accum_prob: np.array, 一维累计概率表
        """
        accum_fitness = fitness.copy()
        for i in range(len(fitness) - 1):
            accum_fitness[i] = np.sum(fitness[: i + 1])
        accum_fitness[-1] = np.sum(fitness)
        accum_prob = accum_fitness / (accum_fitness[-1])
        return accum_prob

    def _gen_children(self, accum_prob):
        r"""
        根据累积概率表生成同样规模的子代种群
        :param accum_prob:
        """
        rand_nums = np.random.random(self.pop_size)
        children_nums = []
        for i in range(len(rand_nums)):
            num = rand_nums[i]

            if num <= accum_prob[0]:
                children_nums.append(0)
            else:
                for j in range(len(accum_prob) - 1):
                    if accum_prob[j] < num <= accum_prob[j + 1]:
                        children_nums.append(j + 1)
        self.pop = self.pop[children_nums, :]

    def _cross_over(self):
        """交配, 相邻样本随机交换元素"""
        for i in range(self.pop_size - 1):
            if np.random.random() < self.pc:
                cpoint = np.random.randint(0, self.chrome_len)
                self.pop[i, cpoint], self.pop[i + 1,
                                              cpoint] = self.pop[i + 1, cpoint], self.pop[i, cpoint]

    def _mutate(self):
        """突变，随机样本随机位点突变"""
        for i in range(self.pop_size):
            if np.random.random() < self.pm:
                mpoint = np.random.randint(0, self.chrome_len)
                bounds, type = self.chrome_bounds[mpoint], self.chrome_types[mpoint]
                if type == 1:
                    self.pop[i, mpoint] = np.random.random(
                    ) * (bounds[1] - bounds[0]) + bounds[0]
                elif type == 0:
                    self.pop[i, mpoint] = np.random.choice(bounds)

    def _best_individual(self, fitness):
        """获取最优个体和对应的适应度"""
        best_fitness = np.max(fitness)
        best_individual = self.pop[np.argmax(fitness), :].reshape(
            1, -1)[0, :]  # **防止有多个最优解
        return best_fitness, best_individual

    @time_cost
    def evolution(self, obj_func, optim_direc: str, epochs: int, max_no_change: int, normalize: bool = True,
                  fit_adj_func=_exp_adj_func, verbose: bool = True) -> Tuple[float, list, list]:
        """
        执行进化
        :param fit_adj_func: func, 函数对象, 用于对适应度进行非线性调节以计算对应概率
        :param normalize: bool, 是否对fitness值进行min-max归一化
        :param obj_func: function, 优化目标函数
                * 如果待优化问题包含约束条件, 则写成lambda x: obj_func(x) + constr_func(x)的形式
                * 约束条件函数必须为 f(x) = 0的隐函数格式, 且使用python基本运算编写, 否则符号函数运算容易报错
        :param optim_direc: str from {'minimize', 'maximize'}, 优化方向
        :param epochs: int, 优化步数
        :param max_no_change: int, 最长无改变次数
        :param verbose: bool, 是否打印学习过程
        :return:
                final_fitness: float, 最终适应度
                final_individual: list of floats or ints, 最终筛选出的个体
                eval_process: list, 进化过程记录

        Example:
        _____________________________________________________________
        # 目标函数
        def min_obj_func(x):
                return np.linalg.norm(x, 2)

        # 设定参数
        chrome_len = 3
        chrome_bounds = [[-1, 0], [0.5, 1, 2, 3], [-1.1, 1, 1.2]]
        chrome_types = [1, 0, 0]
        pop_size = 100
        pc = 0.4
        pm = 0.2
        optim_direc = 'minimize'
        epochs = 2000

        # 进行优化
        ga = GeneticAlgorithm(chrome_len, chrome_bounds, chrome_types, pop_size, pc, pm)
        final_fitness, final_individual, _ = ga.evolution(
                min_obj_func,
                optim_direc = optim_direc,
                epochs = epochs
        )
        _____________________________________________________________
        """
        eval_process = []
        global_best_fitness, global_best_individual = None, None
        no_change = 0
        for epoch in range(epochs):

            # 计算当前最优适应度和最优个体
            fitness, nmlzd_fitness = self._cal_fitness(
                obj_func, optim_direc, normalize)
            best_fitness, best_individual = self._best_individual(fitness)

            if epoch == 0:
                global_best_fitness, global_best_individual = best_fitness, best_individual.copy()
            else:
                if best_fitness > global_best_fitness:
                    global_best_fitness, global_best_individual = best_fitness, best_individual.copy()

            # 记录进化过程
            if verbose:
                print('epoch: {}, global_best_fitness: {:.6f}'.format(
                    epoch, global_best_fitness))
            eval_process.append(
                [global_best_fitness, list(global_best_individual)])

            # 执行遗传、交配和变异
            adj_fitness = fit_adj_func(nmlzd_fitness)
            accum_prob = self._get_accum_prob(adj_fitness)
            self._gen_children(accum_prob)
            self._cross_over()
            self._mutate()

            # 查看优化结果是否有明显改变
            if epoch > 0:
                if eval_process[-1][0] == eval_process[-2][0]:
                    no_change += 1
                else:
                    no_change = 0

            if no_change == max_no_change:
                print('Reaching the max_no_change number, break the loop')
                break

        final_fitness, final_individual = eval_process[-1]

        return final_fitness, final_individual, eval_process
