# coding: utf-8
# Author: lichenarthurdata@gmail.com
from nose.tools import *
import sys
sys.path.append('../')
from lib.math_lib import *

class Test(object):
	def test_square(self):
		assert_equal(math_square(4), 16)
		assert_equal(math_square(-4), 16)
		assert_equal(math_square(0), 0)