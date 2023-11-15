# coding: utf-8
# Author: lichenarthurdata@gmail.com
import sys
import json
import logging
import numpy as np
sys.path.append('../')

from lib.config_loader import config
from lib.math_lib import math_square, math_cube
config.set_logging()


_logger = logging.getLogger(__name__)

from flask import Flask, request, Response, jsonify
app = Flask(__name__)


@app.route('/hello/', methods=['GET'])
def api_hello():
	return json.dumps({'code': 0, 'message': 'successfully', 'data': {'result': 'Hello World!'}})


@app.route('/square/', methods=['PUT'])
def api_square():
	_logger.info('square is requested')
	data = request.values
	try:
		value = int(data['value'])
		result = math_square(value)
		return json.dumps({'code': 0, 'message': '计算成功', 'data': {'result': result}})
	except Exception as e:
		_logger.exception(e)
		return json.dumps({'code': 1, 'message': '计算失败', 'data': {}})


@app.route('/cube/', methods=['POST'])
def api_cube():
	_logger.info('cube is requested')
	data = request.values
	try:
		value = int(data['value'])
		result = math_cube(value)
		return json.dumps({'code': 0, 'message': '计算成功', 'data': {'result': result}})
	except Exception as e:
		_logger.exception(e)
		return json.dumps({'code': 1, 'message': '计算失败', 'data': {}})


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000, debug=False, threaded=False, processes=1)



















