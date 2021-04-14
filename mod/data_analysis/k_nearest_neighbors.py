# -*- coding: utf-8 -*-
"""
Created on 2020/1/22 下午4:42

@Project -> File: algorithm-tools -> k_nearest_neighbors.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 寻找K近邻
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np


class CalMutualInfluences(object):
    """
    使用矩阵方法计算list内部元素间两两欧式距离.
    优化了高维数据计算效率和内存占用, 但计算效率还是不如树方法
    """

    def __init__(self, array):
        """
        初始化
        :param array: 待计算的array, 待计算的元素按列排列
        """

        self.array = array.T
        self.shape = array.T.shape
        self.sample_dim = self.shape[0]
        self.sample_num = self.shape[1]

    def cal_matrix_C(self):
        """生成中间矩阵C"""
        self.C = []
        for i in range(self.sample_num):
            m = np.zeros([self.sample_num, 1])
            m[i] = 1
            m = np.eye(self.sample_num, self.sample_num) - \
                m * np.ones(self.sample_num)
            self.C.append(m)

    def cal_matrix_B(self):
        """计算矩阵B"""
        self.B = list()
        for i in range(self.sample_num):
            self.B.append(np.dot(self.array, self.C[i]))
        del self.C

    def cal_results(self):
        self.cal_matrix_C()
        self.cal_matrix_B()
        results = list()
        for i in range(self.sample_num):
            results.append(np.diag(np.dot(self.B[i].T, self.B[i])))
        results = np.power(np.array(results).reshape(
            self.sample_num, self.sample_num), 0.5)
        del self.B
        return results


class KNearestNeighbors(object):
    """
    样本中K近邻计算
    """

    def __init__(self, X, n_neighbors):
        """
        初始化.
        :param X: np.ndarray, 样本点坐标矩阵
        :param n_neighbors: int > 0, 指定近邻点个数
        """

        self.X = X
        self.n_neighbors = n_neighbors

    def cal_neighbors_indices_and_distances(self, method='tree', **kwargs):
        """
        计算近邻点的编号和距离.
        :param method: str, 选用计算方法: {'matrix', 'tree'}
        :param kwargs:
                'tree_algorithm': 不指定或者参考sklearn.neighbors.NearestNeighbors中algorithm的设置
        :return: distances: np.ndarray, 距离矩阵
        :return: indices: np.ndarray, 索引矩阵

        Example:
        ——————————————————
        import pandas as pd

        # 载入数据
        data = pd.read_csv('../data/device_info.csv')

        # 数据格式转换, 每个测量值对应的X为[lon, lat, time]
        value_cols = [p for p in data.columns if 'value_' in p]

        data_new = None
        for col in value_cols:
                data_piece = data[['deviceID', 'name', 'long', 'lat', col]].copy()
                data_piece['time'] = data_piece.apply(lambda x: int(col.split(sep = '_')[1]), axis = 1)
                data_piece.rename(columns = {col: 'value'}, inplace = True)

                if value_cols.index(col) == 0:
                        data_new = data_piece
                else:
                        data_new = pd.concat([data_new, data_piece], axis = 0, sort = False)

        data_new.reset_index(drop = True, inplace = True)

        # 整理数据, 并对不同维度进行加权
        name_cols = ['deviceID', 'name']
        loc_cols = ['long', 'lat', 'time']
        value_cols = ['value']
        weights = {'long': 1, 'lat': 1, 'time': 10}
        data_new = data_new[name_cols + loc_cols + value_cols]
        for col in weights.keys():
                data_new[col] = data_new[col].apply(lambda x: x * weights[col])

        # K近邻计算
        X = np.array(data_new[loc_cols])
        knn = KNearestNeighbors(X, n_neighbors = 20)
        distances, indices = knn.cal_neighbors_indices_and_distances(method = 'tree')
        ——————————————————
        """
        if method == 'matrix':
            if self.X.shape[0] * self.X.shape[1] > 500:
                print(
                    'Warning: low efficiency may arise with method = "matrix", you should better switch to method = "tree"')

            cmf = CalMutualInfluences(self.X)
            dist_matrix = cmf.cal_results()  # 计算站点间的距离
            print('finish cmf calculation')

            distances = np.sort(dist_matrix, axis=1)[:, :self.n_neighbors]
            indices = np.argsort(dist_matrix, axis=1)[:, :self.n_neighbors]
            return distances, indices
        elif method == 'tree':
            if 'tree_algorithm' not in kwargs.keys():
                algorithm = 'ball_tree'
            else:
                algorithm = kwargs['tree_algorithm']

            nbrs = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                algorithm=algorithm
            ).fit(self.X)
            distances, indices = nbrs.kneighbors(self.X)
            return distances, indices
        else:
            raise ValueError('unknown method "{}"'.format(method))
