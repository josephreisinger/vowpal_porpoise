cimport vw_py

cdef class VW:
    def __init__(self, params):
        vw_py.initialize(params)

    def learn(self, example):
        return vw_py.learn(example)

    def finish(self):
        vw_py.finish()
