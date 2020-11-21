"""Misc Utilities"""

def sizeof_fmt(num, suffix='B'):
    """
    Change size from bytes to human readable unit
    Args:
        num (Int): Size in bytes
        Suffix (Str): Bytes or B or any suffix you wish
    """

    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0

    return "%.1f %s%s" % (num, 'Y', suffix)
