import multiprocessing

bind = '0.0.0.0:8000'
workers = multiprocessing.cpu_count() if multiprocessing.cpu_count() <= 2 else 2
threads = 1
daemon = True
pidfile = '../gunicorn.pid'

def post_worker_init(worker):
	pass
