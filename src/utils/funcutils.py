"""Function Utils, decorators for debugging and other utilities"""
import functools
import gc

def debug(func):
    """Print the function signature and return value"""

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper_debug


def debug_return(func):
    """Print the function name & return value"""

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        print(f"Calling {func.__name__}()")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper_debug


def debug_signature(func):
    """Print the function signature"""

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned")
        return value
    return wrapper_debug


def force_garbage_collection(before, after):
    """
    Force garbage collection before & after the function execution
    Args:
        before (Bool): Force garbage collection before the function execution
        after (Bool): Force garbage collection after the function execution
    """
    def decorator_gc(func):
        @functools.wraps(func)
        def wrapper_gc(*args, **kwargs):
            if before:
                gc.collect()

            value = func(*args, **kwargs)

            if after:
                gc.collect()
            return value
        return wrapper_gc
    return decorator_gc
