import time

ENABLE_TIMING = False


# implement a timing decorator
def time_it(func):
    if ENABLE_TIMING:

        def wrap(*args, **kwargs):
            time1 = time.time()
            ret = func(*args, **kwargs)
            time2 = time.time()
            print(f"function {func.__name__} took {time2 - time1} s")
            return ret

        return wrap
    else:
        return func
