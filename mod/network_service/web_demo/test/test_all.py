# coding: utf-8
# 测试所有

import lake.shell

if __name__ == '__main__':

	# 删除项目中的所有.pyc缓存
	# 如果不删除，单元测试可能使用的.pyc文件，而.py文件已经被干掉了
	lake.shell.run('find ../ -name "*.pyc" | xargs rm -rf')

	# 测试
	lake.shell.run('nosetests -v')