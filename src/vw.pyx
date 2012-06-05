cdef extern from "vw_py.h":
    cdef cppclass vw_py:
        vw_py(char*)
        float learn(char*)
