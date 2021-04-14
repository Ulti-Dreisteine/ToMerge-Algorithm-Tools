# -*- coding: utf-8 -*-
"""
Created on 2020/2/20 15:36

@Project -> File: algorithm-tools -> request_task_op.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 网络操作
"""

import requests
import json

req_ops_available = ['pull_task_info', 'pull_task_data', 'call_back']


class RequestOperation(object):
	"""请求操作"""
	
	def __init__(self, task_id = None, req_op = None, task_type = None, project_type = None, data = None):
		"""
		初始化
		:param task_id: str, 任务id, op_type = 'pull_task_data' 或 'call_back'时使用
		:param op_type: str, 操作类型, 必须在{'pull_task_info', 'pull_task_data', 'call_back'}中
		:param task_type: str, 任务类型, op_type = 'pull_task_info'时使用
		:param project_type: str, 项目类型, op_type = 'pull_task_data'时使用, 具体值需要查询对应接口文档
		:param data: dict, 待返回数据,
			计算成功时 data = {'progress': 'SUCCESS', 'remark': 'Success', 'result': results}
			计算失败时 data = {'progress': 'FAILED', 'remark': '**错误信息**', 'result': {}}
		"""
		# 根据请求操作类型检查参数.
		try:
			assert req_op in req_ops_available
		except:
			raise ValueError('ERROR: req_op {} not in req_ops_available {}'.format(req_op, req_ops_available))
		
		if req_op == 'pull_task_info':
			try:
				assert task_type is not None
			except:
				raise ValueError('ERROR: task_type is None.')
		elif req_op == 'pull_task_data':
			try:
				assert task_id is not None
				assert project_type is not None
			except:
				raise ValueError('ERROR: task_id or project_type is None')
		elif req_op == 'call_back':
			try:
				assert task_id is not None
				assert data is not None
			except:
				raise ValueError('ERROR: task_id or data is None')
		else:
			pass
		
		self.task_id = task_id
		self.req_op = req_op
		self.task_type = task_type
		self.project_type = project_type
		self.data = data
	
	def _build_url(self, root_url):
		"""
		通过任务参数构建请求url地址
		:param root_url: str of url, 请求根地址, 通过环境变量读取
		"""
		if self.req_op == 'call_back':
			url = root_url + str(self.task_id) + '/feedback'  # call_back_url/{request_id}/feedback
		elif self.req_op == 'pull_task_info':
			url = root_url + self.task_type
		elif self.req_op == 'pull_task_data':
			url = root_url + str(self.task_id) + '/{}'.format(self.project_type)
		else:
			raise RuntimeError('ERROR: cannot build url for the request.')
		return url
		
	def request(self, root_url):
		"""
		进行请求
		:param root_url: str of url, 请求根地址
		
		Example:
		------------------------------------------------------------
		# %% 拉取任务信息.
		req_op = 'pull_task_info'
		task_type = 'STD_POLLUTION_FORECAST'
		root_url = 'http://srv-iep-ai-task__intelliep.rocktl.com/iep-ai-task/v0/tasks/'
		ro = RequestOperation(req_op = req_op, task_type = task_type)
		resp_data = ro.request(root_url)
		task_id = ro.summerize_results(resp_data, return_task_id = True)
		
		# %% 拉取任务数据.
		req_op = 'pull_task_data'
		task_id = '000000000000000'
		root_url = 'http://srv-iep-ai-task__intelliep.rocktl.com/iep-ai-task/v0/data/'
		ro = RequestOperation(req_op = req_op, task_id = task_id, project_type = 'pollutionForecast')
		resp_data = ro.request(root_url)
		
		# %% 回调.
		req_op = 'call_back'
		task_id = '000000000000000'
		data = {}
		root_url = 'http://srv-iep-ai-task__intelliep.rocktl.com/iep-ai-task/v0/tasks/'
		ro = RequestOperation(req_op = req_op, task_id = task_id, data = data)
		resp_data = ro.request(root_url)
		------------------------------------------------------------
		"""
		url = self._build_url(root_url)
		retry = 0
		while True:
			if self.req_op in ['pull_task_info', 'pull_task_data']:
				resp = requests.get(url)
			elif self.req_op in ['call_back']:
				resp = requests.put(url, json = self.data)
			else:
				raise ValueError('ERROR: invalid self.req_op {}'.format(self.req_op))
			
			if retry < 3:
				if resp.status_code >= 500:
					# 重试请求.
					print('Retry requesting, retry = {}'.format(retry))
					retry += 1
					continue
				elif resp.status_code in [200, 204]:
					break
			else:
				raise RuntimeError('ERROR: reach max request time = 3, cannot get response, req_op = {}'.format(self.req_op))
		
		resp_data = json.loads(resp.text)
		return resp_data
	
	def summerize_results(self, resp_data, **kwargs):
		"""
		整理请求结果
		:param resp_data: dict, 请求获得的数据
		:param kwargs:
			return_task_id: bool, 用于self.req_op = 'pull_task_info'时, if True, 返回task_id, 否则返回resp_data
		"""
		if self.req_op == 'pull_task_info':
				if 'return_task_id' in kwargs.keys():
					if kwargs['return_task_id']:
						return resp_data['taskID']
					else:
						return resp_data
				else:
					return resp_data
		elif self.req_op == 'pull_task_data':
			return resp_data
		else:
			raise RuntimeError('ERROR: cannot summerize request results.')
		

if __name__ == '__main__':
	pass
	
	
	
		
		


