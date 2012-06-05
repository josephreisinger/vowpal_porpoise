cimport vw_py

cdef class VW:
    def __init__(self, params):
        vw_py.initialize(params)

    def learn(self, example):
        vw_py.learn(example)

    def destory(self):
        vw_py.finalize()


