import functools
import os


def safe_remove(f):
    try:
        os.remove(f)
    except OSError:
        pass

"""
memoized.py

Simple wrapper class for memoizing a function, stores the args that the function
was called with mapped to the return value. Won't check for nondeterminism, so
don't use this with probabilistic methods.

default is just like:

@memoized
def factorial(...

Or you could just use the python3 builtin:
from funtools import lru_cache

@lru_cache(1000)
def factorial(...

Memoized also works on class/instance methods, so you can do:
class Foo:
  @memoized
  def f(self, x):
     ...
"""


class memoized(object):
    def __init__(self, f):
        self.f = f
        self.m = {}

    def __get__(self, instance, owner=None):
        if instance is None:
            # f is being accessed as a class method (e.g. C.f)
            return self.f
        else:
            # f is being called as an instance method; when we call f, its self
            # argument should be set to instance.
            return functools.partial(self, instance)

    def __call__(self, *args):
        if not args in self.m:
            self.m[args] = self.f(*args)
        return self.m[args]


"""
Basic logger functionality; replace this with a real logger of your choice
"""


class VPLogger:
    def debug(self, s):
        print '[DEBUG] %s' % s

    def info(self, s):
        print '[INFO] %s' % s

    def warning(self, s):
        print '[WARNING] %s' % s

    def error(self, s):
        print '[ERROR] %s' % s
