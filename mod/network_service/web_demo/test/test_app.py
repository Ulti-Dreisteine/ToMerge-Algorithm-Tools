# coding: utf-8
# Author: lichenarthurdata@gmail.com
from nose.tools import *
import urllib
import json
import sys
sys.path.append('../')
from bin.app import *


def api_get(path, data):
	url = '%s?%s' % (path, urllib.parse.urlencode(data))
	print('-' * 100)
	print('请求接口: ' + url)
	c = app.test_client()
	response = c.get(url)
	res_obj = json.loads(response.data.decode('utf-8'))
	return res_obj

def api_put(path, data):
	print('-' * 100)
	print('请求接口: ' + path)
	c = app.test_client()
	response = c.put(path, data=data)
	print(response.data)
	res_obj = json.loads(response.data.decode('utf-8'))
	return res_obj

def api_post(path, data):
	print('-' * 100)
	print('请求接口: ' + path)
	c = app.test_client()
	response = c.post(path, data=data)
	print(response.data)
	res_obj = json.loads(response.data.decode('utf-8'))
	return res_obj

def api_hello(req_dict):
	return api_get('/hello/', req_dict)

def api_square(req_dict):
	return api_put('/square/', req_dict)

def api_cube(req_dict):
	return api_post('/cube/', req_dict)

class Test(object):
	def test_api_hello(self):
		res = api_hello({})
		assert_equal(res['code'], 0)
		assert_equal(res['data']['result'], 'Hello World!')

	def test_api_square(self):
		value = 4
		res = api_square({'value': value})
		assert_equal(res['code'], 0)
		assert_equal(res['data']['result'], 16)

	def test_api_cube(self):
		value = 4
		res = api_cube({'value': value})
		assert_equal(res['code'], 0)
		assert_equal(res['data']['result'], 64)


if __name__ == '__main__':
	test = Test()
	test.test_api_test()
	test.test_api_square()
	test.test_api_cube()












