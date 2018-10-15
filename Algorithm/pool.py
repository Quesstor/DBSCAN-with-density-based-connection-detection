import multiprocessing


pool = False
def getThreadPool():
    global pool
    if pool: return pool
    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 4
    pool = multiprocessing.Pool(processes=cpus)
    return pool