# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

配置workers
"""
import os
import redis
from rq import Worker, Queue, Connection

listen = ['high', 'default', 'low']  # 监听队列名, 注意顺序
redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)


if __name__ == '__main__':
    with Connection(conn):
        # 创建worker
        worker = Worker(map(Queue, listen), default_worker_ttl = 1)  # TODO: default_worker_ttl参数意义？

        #  启动worker
        worker.work()


