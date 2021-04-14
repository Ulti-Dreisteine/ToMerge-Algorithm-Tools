# -*- coding: utf-8 -*-
"""
Created on 2020/2/21 14:47

@Project -> File: realtime-wind-rose-diagram -> request_device_op.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 
"""

import pandas as pd
import requests
import json

from ..decorator import timeout

req_ops_available = ['pull_device_value', 'pull_device_ids_list', 'pull_nearby_device_ids_list']


class RequestOperation(object):
	"""请求操作"""
	
	def __init__(self, req_op: str, params: dict = None):
		"""
		初始化
		"""
		try:
			assert req_op in req_ops_available
		except:
			raise ValueError('req_op {} not in req_ops_available {}'.format(req_op, req_ops_available))
		
		if req_op == 'pull_device_value':
			try:
				assert params is not None
			except:
				raise ValueError('params is None.')
		elif req_op == 'pull_device_ids_list':
			try:
				assert params is not None
				for key in ['device_type', 'project_id']:
					assert key in params.keys()
			except Exception as e:
				raise ValueError('params is None or some keys are not in params.keys(), {}'.format(e))
			params = {
				'deviceType': params['device_type'],
				'projectID': params['project_id']
			}
			
		elif req_op == 'pull_nearby_device_ids_list':
			try:
				assert params is not None
				for key in ['lon', 'lat', 'device_type', 'distance', 'project_id']:
					assert key in params.keys()
			except Exception as e:
				raise ValueError('params is None or some keys are not in params.keys(), {}'.format(e))
			
			params = {
				'lat': params['lat'],
				'lon': params['lon'],
				'deviceType': params['device_type'],
				'distance': params['distance'],
				'projectID': params['project_id']
			}
			
		else:
			pass
		
		self.req_op = req_op
		self.params = params
		
	def _build_url(self, root_url):
		url = None
		if self.req_op == 'pull_device_value':
			url = root_url
		elif self.req_op == 'pull_device_ids_list':
			url = root_url
		elif self.req_op == 'pull_nearby_device_ids_list':
			url = root_url + '{}/{}'.format(self.params['lon'], self.params['lat'])
		else:
			pass
		return url
	
	@timeout(60)
	def request(self, root_url):
		try:
			url = self._build_url(root_url)
			retry = 0
			while True:
				if self.req_op in ['pull_device_value', 'pull_device_ids_list']:
					resp = requests.get(url, params = self.params)
				elif self.req_op in ['pull_nearby_device_ids_list']:
					resp = requests.get(url, params = self.params)
				else:
					raise ValueError('Invalid self.req_op {}'.format(self.req_op))
				
				if retry < 3:
					if resp.status_code >= 500:
						# 重试请求.
						print('Retry requesting, retry = {}'.format(retry))
						retry += 1
						continue
					elif resp.status_code in [200, 204]:
						break
				else:
					raise RuntimeError(
						'Reach max request time = 3, cannot get response, req_op = {}'.format(self.req_op))
			
			data = pd.DataFrame(json.loads(resp.text)['data'])
			return data
		except Exception as e:
			raise RuntimeError('Connection error, {}'.format(e))


if __name__ == '__main__':

	# ---- 测试pull_device_ids_list ----------------------------------------------------------------
	
	from lib import env_params
	from lib import PROJECT_IDS
	project_id = PROJECT_IDS['relevance-analyzer']
	params = {'device_type': 'VEHICLE', 'project_id': project_id}
	url = env_params['GetDeviceIdsList']
	ro = RequestOperation(req_op = 'pull_device_ids_list', params = params)
	data = ro.request(url)