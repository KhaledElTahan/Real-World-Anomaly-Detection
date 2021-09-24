"""Function Utils, decorators for debugging and other utilities"""
import functools
import gc
import cProfile
import io
import pstats
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


def profile(apply=False, lines_to_print=None, strip_dirs=False):
    """
    Profile the function & print stats
    Args:
        apply (Bool): Turn on or off the effect of this decorator
        lines_to_print (int | None): Number of lines to print. Default (None) is for all the lines.
        strip_dirs (bool): Whether to remove the leading path info from file names.
    """
    def decorator_pr(func):
        @functools.wraps(func)
        def wrapper_pr(*args, **kwargs):
            if apply:
                func_profile = cProfile.Profile()
                func_profile.enable()

            retval = func(*args, **kwargs)

            if apply:
                func_profile.disable()

                stream_str = io.StringIO()
                profile_stats = pstats.Stats(func_profile, stream=stream_str)

                if strip_dirs:
                    profile_stats.strip_dirs()

                profile_stats.sort_stats(pstats.SortKey.CUMULATIVE)
                profile_stats.print_stats(lines_to_print)

                print(stream_str.getvalue())

            return retval
        return wrapper_pr
    return decorator_pr
