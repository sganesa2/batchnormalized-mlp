from functools import wraps
import time

def timeit(func):
    @wraps
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end-start:.6f}s to run")
        return result
    return wrapper