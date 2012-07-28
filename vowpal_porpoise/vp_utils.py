def safe_remove(f):
    import os
    try:
        os.remove(f)
    except OSError:
        pass

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
