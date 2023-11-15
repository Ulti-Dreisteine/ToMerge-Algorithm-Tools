# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei


"""
from rq import Queue
import time
from redis import Redis
from jobs import func_low, func_high, func_default

redis_conn = Redis()

q_low = Queue('low', connection = redis_conn)
q_high = Queue('high', connection = redis_conn)
q_default = Queue('default', connection = redis_conn)

# 添加任务，注意添加顺序
max_queued_time = 30  # job停留在队列里的最长时间
result_ttl = 100  # 结果保留的最长时间

job_high = q_high.enqueue(
    f = func_high,
    ttl = max_queued_time,
    args = ('http://www.baidu.com',),  # func里的args参数
    result_ttl = result_ttl
)

job_default = q_default.enqueue(
    f = func_default,
    ttl = max_queued_time,
    args = ('http://www.baidu.com',),  # func里的args参数
    result_ttl = result_ttl
)

job_low = q_low.enqueue(
    f = func_low,
    ttl = max_queued_time,
    args = ('http://www.baidu.com',),  # func里的args参数
    result_ttl = result_ttl
)

print('\nstep = 0')
print(job_low.result, job_default.result, job_high.result)

for step in range(1, 200):
    time.sleep(1)
    print('\nstep = {}'.format(step))
    print(job_high.result, job_low.result, job_default.result)  # 即使没有打印也会有后台运行


