# -*- coding: utf-8 -*-
"""
Created on 2019/12/24 下午12:27

@Project -> File: guodian-desulfuration-optimization -> genetic_algorithm.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 粒子群优化算法
"""

from lake.decorator import time_cost
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


@time_cost
class PSO(object):
    """
    粒子群优化算法， 适用于高维、非线性过程的连续变量优化

    Note:
            1. PSO优化要求输入X必须是连续数值型变量, 输出y不作要求
            2. 目前该PSO算法无法处理含约束优化问题
    """

    def __init__(self, obj_func, dim, iter_num, particle_num, dt, c1=2.0, c2=2.0, w=0.8):
        """
        初始化对象
        :param obj_func: 目标函数, 接受一维向量list或np.array输入
        :param dim: int, 优化问题的维数
        :param iter_num: int, 算法迭代的次数
        :param particle_num: int, 算法中用于计算的颗粒数目
        :param dt: float，时间步长
        :param c1: float, 局部学习因子
        :param c2: float, 全局学习因子
        :param w: float, 惯性权重
        """

        self.obj_func = obj_func
        self.dim = dim
        self.iter_num = iter_num
        self.particles_num = particle_num
        self.dt = dt
        self.c1 = c1
        self.c2 = c2
        self.w = w

    def set_init_loc_bounds(self, init_loc_bounds):
        """设置颗粒位置loc在各方向上的上下界, 嵌套list"""
        self.init_loc_bounds = init_loc_bounds

    def set_init_vel_bounds(self, init_vel_bounds):
        """设置颗粒速度vel在各方向上的上下界, 嵌套list"""
        self.init_vel_bounds = init_vel_bounds

    def _init_particle_locs(self):
        """初始化颗粒位置, 各列对应各个维度"""
        self.particle_locs = None
        for i in range(self.dim):
            bounds = self.init_loc_bounds[i]
            locs = np.random.random(self.particles_num) * \
                (bounds[1] - bounds[0]) + bounds[0]

            if i == 0:
                self.particle_locs = locs.reshape(-1, 1)
            else:
                self.particle_locs = np.hstack(
                    (self.particle_locs, locs.reshape(-1, 1)))

    def _init_particle_vels(self):
        """初始化颗粒位置, 各列对应各个维度"""
        self.particle_vels = None
        for i in range(self.dim):
            bounds = self.init_vel_bounds[i]
            vels = np.random.random(self.particles_num) * \
                (bounds[1] - bounds[0]) + bounds[0]

            if i == 0:
                self.particle_vels = vels.reshape(-1, 1)
            else:
                self.particle_vels = np.hstack(
                    (self.particle_vels, vels.reshape(-1, 1)))

    @staticmethod
    def current_particles_locs_and_values(particle_locs, obj_func):
        """计算当前所有颗粒位置和对应目标函数值"""
        current_particles_info = particle_locs.copy()
        current_particles_info = pd.DataFrame(current_particles_info)
        current_particles_info['obj_value'] = current_particles_info.apply(
            obj_func, axis=1)
        return current_particles_info

    @staticmethod
    def global_best_solutions_and_value(particles_info, dim):
        """计算当前全局最佳解"""
        particles_info = particles_info.copy()
        current_g_best_idx = particles_info['obj_value'].idxmin(axis=1)
        g_best_value = particles_info.loc[current_g_best_idx, 'obj_value']
        g_best_solution = np.array(
            particles_info.loc[current_g_best_idx, range(dim)])  # 此刻全局最优解
        return g_best_solution, g_best_value

    def optimization(self, w_decay=None, show_loss=False):
        """
        执行粒子群优化

        Example:
        ——————————————————
        def objective(x):
                return np.linalg.norm(x - 0.2, 2) + np.linalg.norm(np.sin(x), 2)

        dim = 3
        iter_num = 500
        particles_num = 30
        dt = 1.0
        max_vel = 0.2

        pso = PSO(objective, dim, iter_num, particles_num, dt, max_vel)

        # 初始化位置和速度的边界
        init_loc_bounds = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        pso.set_init_loc_bounds(init_loc_bounds)

        init_vel_bounds = [[-max_vel, max_vel], [-max_vel, max_vel], [-max_vel, max_vel]]
        pso.set_init_vel_bounds(init_vel_bounds)

        # 执行pso优化
        g_best_solution, g_best_value, (particles_locus_record, loss) = pso.optimization(w_decay = 0.99, show_loss = True)
        ——————————————————
        """

        # 初始化颗粒位置和速度
        self._init_particle_locs()
        self._init_particle_vels()

        # 计算当前所有颗粒位置对应函数值
        current_particles_info = self.current_particles_locs_and_values(
            self.particle_locs, self.obj_func)

        # 初始化全局最优解和值
        g_best_solution, g_best_value = self.global_best_solutions_and_value(
            current_particles_info, self.dim)

        # 当前每个点的最优解表
        p_best_info = current_particles_info

        # 进行pso计算
        loss = [g_best_value]
        particles_locus_record = [self.particle_locs.copy()]
        for step in range(self.iter_num):
            # 速度更新
            if w_decay is not None:
                self.w *= w_decay

            self.particle_vels += self.w * self.particle_vels
            self.particle_vels += self.c1 * np.random.random() * (
                np.array(p_best_info.loc[:, range(self.dim)]) - self.particle_locs.copy())
            self.particle_vels += self.c2 * np.random.random() * (
                np.dot(np.ones([self.particles_num, 1]), g_best_solution.reshape(1, -1)) - self.particle_locs.copy())

            # 各个维度上的最大最小速度限制
            for i in range(len(self.init_vel_bounds)):
                self.particle_vels[:, i][self.particle_vels[:, i] >
                                         self.init_vel_bounds[i][1]] = self.init_vel_bounds[i][1]
                self.particle_vels[:, i][self.particle_vels[:, i] <
                                         self.init_vel_bounds[i][0]] = self.init_vel_bounds[i][0]

            # 位置更新
            self.particle_locs += self.particle_vels

            # 计算当前所有颗粒位置函数值, 并更新全局最优解记录
            current_particles_info_tmp = self.current_particles_locs_and_values(
                self.particle_locs, self.obj_func)
            g_best_solution_tmp, g_best_value_tmp = self.global_best_solutions_and_value(
                current_particles_info_tmp, self.dim)

            if g_best_value_tmp < g_best_value:
                g_best_solution, g_best_value = g_best_solution_tmp.copy(), g_best_value_tmp

            # 更新各点的最优点信息
            for row in range(self.particles_num):
                if current_particles_info_tmp.loc[row, 'obj_value'] < current_particles_info.loc[row, 'obj_value']:
                    current_particles_info.loc[row,
                                               :] = current_particles_info_tmp.loc[row, :]

            loss.append(g_best_value)
            particles_locus_record.append(self.particle_locs.copy())

            print('iter: {}, loss: {:.6f}, current best solution: {}'.format(
                step, g_best_value, g_best_solution))

        if show_loss:
            print('优化过程画图, 最多显示前两维的变化情况')
            plt.figure(figsize=[6, 10])
            if self.dim == 1:
                plt.subplot(2, 1, 1)
                plt.plot(loss, linewidth=2.0)
                plt.xlabel('epochs')
                plt.ylabel('loss')

                plt.subplot(2, 1, 2)
                for i in range(self.particles_num):
                    plt.plot(
                        [r[i, 0] for r in particles_locus_record],
                        linewidth=0.5,
                        alpha=0.4
                    )
                plt.xlabel('epochs')
                plt.ylabel('param value')
            else:
                plt.subplot(2, 1, 1)
                plt.plot(loss, linewidth=2.0)
                plt.xlabel('epochs')
                plt.ylabel('loss')

                plt.subplot(2, 1, 2)
                for i in range(self.particles_num):
                    plt.plot(
                        [r[i, 0] for r in particles_locus_record], [r[i, 1]
                                                                    for r in particles_locus_record],
                        linewidth=0.5,
                        alpha=0.4
                    )
                plt.scatter(
                    g_best_solution[0], g_best_solution[1], c='k', s=80, alpha=1.0, label='optimal')
                plt.legend(loc='upper right')
                plt.xlabel('longitude')
                plt.ylabel('latitude')

        return g_best_solution, g_best_value, (particles_locus_record, loss)
