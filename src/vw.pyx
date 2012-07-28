cimport vw_c

cdef class VW:
    def __init__(self, params):
        vw_c.initialize(params)

    def learn(self, example):
        return vw_c.learn(example)

    def finish(self):
        vw_c.finish()
