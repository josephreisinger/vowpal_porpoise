def safe_remove(f):
    import os
    try:
        os.remove(f)
    except OSError:
        pass

"""
Basic logger functionality; replace this with a real logger of your choice
"""
import imp
import sys


class VPLogger:
    def debug(self, s):
        print '[DEBUG] %s' % s

    def info(self, s):
        print '[INFO] %s' % s

    def warning(self, s):
        print '[WARNING] %s' % s

    def error(self, s):
        print '[ERROR] %s' % s


def import_non_local(name, custom_name=None):
    """Import when you have conflicting names"""
    custom_name = custom_name or name

    f, pathname, desc = imp.find_module(name, sys.path[1:])
    module = imp.load_module(custom_name, f, pathname, desc)
    if f:
      f.close()

    return module
