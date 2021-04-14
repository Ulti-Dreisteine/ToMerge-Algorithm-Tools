# -*- coding: utf-8 -*-
"""
Created on 2020/2/20 16:51

@Project -> File: algorithm-tools -> timeout.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 函数计算超时处理
"""

from threading import Thread
import functools

__doc__ = """
	# 算例.
	import time
	from lake.decorator import time_cost

	@time_cost
	@timeout(1)
	def test_func(k = 0):
		time.sleep(5)
		k += 1
		return k
	
	k = 0
	k = test_func(k)
	
	print(k)
"""


def timeout(timeout):
	"""
	超时监测
	**注意装饰函数中不能出现print等语句
	:param timeout: int or float, 超时秒数设置
	"""
	
	def deco(func):
		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
			
			def newFunc():
				try:
					res[0] = func(*args, **kwargs)
				except Exception as e:
					res[0] = e
			
			t = Thread(target = newFunc)
			t.daemon = True
			try:
				t.start()
				t.join(timeout)
			except Exception as je:
				print('error starting thread')
				raise je
			ret = res[0]
			if isinstance(ret, BaseException):
				raise ret
			return ret
		
		return wrapper
	
	return deco
