"""Function Utils, decorators for debugging and other utilities"""
import functools
import gc
from pprint import pprint

from src.utils import debugutils


def debug(apply=True, sign=True, ret=True, sign_beautify=False, ret_beautify=False):
    """
    Print the function signature and/or return value
    Args:
        apply (Bool): Turn on or off the effect of this decorator
        sign (Bool): Print function signature
        ret (Bool): Print function return value
        sign_beautify (Bool): Beautify function signature
        ret_beautify (Bool): Beautify function return value
    """
    def decorator_debug(func):
        @functools.wraps(func)
        def wrapper_debug(*args, **kwargs):
            if not apply:
                return func(*args, **kwargs)

            if sign:
                args_repr = [
                    debugutils.tensors_to_shapes(a)
                    if sign_beautify else repr(a)
                    for a in args
                ]
                kwargs_repr = [
                    f"{k}={debugutils.tensors_to_shapes(v)!r}"
                    if sign_beautify else f"{k}={v!r}"
                    for k, v in kwargs.items()]
                if sign_beautify:
                    print(f"Calling {func.__name__}()")
                    if len(args) > 0:
                        print("Arguments:") 
                        pprint(args_repr)
                    if len(kwargs) > 0:
                        print("Keyword Arguments:")
                        pprint(kwargs_repr)
                else:
                    signature = ", ".join(args_repr + kwargs_repr)
                    print(f"Calling {func.__name__}({signature})")
            else:
                print(f"Calling {func.__name__}()")

            print()
            value = func(*args, **kwargs)

            if ret:
                if ret_beautify:
                    print(f"{func.__name__!r} returned")
                    print("Return Value:")
                    pprint(debugutils.tensors_to_shapes(value))
                else:
                    print(f"{func.__name__!r} returned {value!r}")
            else:
                print(f"{func.__name__!r} returned")

            print()

            return value
        return wrapper_debug
    return decorator_debug


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


def static_variables(**wrapper_kwargs):
    """
    Adds set of static variables to methods
    """
    def decorator_sv(func):
        @functools.wraps(func)
        def wrapper_sv(*args, **kwargs):
            for k in wrapper_kwargs:
                setattr(func, k, wrapper_kwargs[k])

            func(*args, **kwargs)
        return wrapper_sv
    return decorator_sv
