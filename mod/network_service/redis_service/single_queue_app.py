# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

单序列redis work
"""
from rq import Connection, Queue, Worker
import time
from redis import Redis
from jobs import count_words_at_url

# Tell RQ what Redis connection to use
redis_conn = Redis()
q = Queue('default', connection = redis_conn)  # no args implies the default queue

# 延迟计算count_words_at_url
job = q.enqueue(count_words_at_url, 'http://nvie.com')
time.sleep(1)
print(job.result)

# 等待，直到worker计算完毕
time.sleep(10)
print(job.result)







