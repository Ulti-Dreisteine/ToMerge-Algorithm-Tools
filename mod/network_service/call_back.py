# -*- coding: utf-8 -*-
"""
Created on 2020/2/25 11:33

@Project -> File: realtime-wind-rose-diagram -> call_back.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 回调
"""

import requests


def call_back(url, data):
	retry = 0
	while True:
		resp = requests.put(url, json = data)
		if retry < 3:
			if resp.status_code >= 500:
				# 重试请求.
				print('Retry requesting, retry = {}'.format(retry))
				retry += 1
				continue
			elif resp.status_code in [200, 204]:
				break
		else:
			raise RuntimeError('ERROR: reach max request time = 3, cannot call back.')
