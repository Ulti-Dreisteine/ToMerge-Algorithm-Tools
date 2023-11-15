# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei


"""

import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn import mixture

warnings.filterwarnings('ignore')


def gen_fake_samples(geo_locs, mean):
	"""
	生成假样本数据
	:param geo_locs:
	:param len:
	:return:
	"""
	func = lambda loc: 10 / (np.linalg.norm(loc - mean, 2) + 0.01)  # 离均值越近值越大
	values = np.array([func(loc) for loc in geo_locs]).reshape(len(geo_locs), 1)
	
	return np.hstack((geo_locs, values))


def gen_data():
	sample_num = 500
	mean = [0, 0]
	cov = [[1, 0], [0, 1]]
	geo_locs = np.random.multivariate_normal(mean = mean, cov = cov, size = sample_num)
	samples = gen_fake_samples(geo_locs, mean)
	mean = [10, 2]
	geo_locs = np.random.multivariate_normal(mean = mean, cov = cov, size = sample_num)
	samples = np.vstack((gen_fake_samples(geo_locs, mean), samples))
	
	return samples


def gaussian_mixture(samples):
	clf = mixture.GaussianMixture(n_components = 5, covariance_type = 'full')
	clf.fit(samples)
	return clf

if __name__ == '__main__':
	# 数据和参数
	samples = gen_data()
	
	clf = gaussian_mixture(samples)
	
	# 进行插值
	x = np.linspace(-3., 15., 50)
	y = np.linspace(-3., 15., 50)
	z = np.linspace(0.0, 2.0, 50)
	X, Y, Z = np.meshgrid(x, y, z)
	XX = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T
	
	Z = clf.score_samples(XX)
	Z = Z.reshape(X.shape)
	
	z_ml = np.zeros((X.shape[0], X.shape[1]))
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			z_ml[i][j] = np.max(Z[i][j][:])
	
	X, Y = np.meshgrid(x, y)
	c = plt.contour(X, Y, z_ml, 30)
	plt.clabel(c, inline = True, fontsize = 4)
	
	probs = clf.predict_proba(XX)
	labels = clf.predict(XX)
	labels = labels.reshape(len(labels), 1)
	XX = np.hstack((XX, labels))
	