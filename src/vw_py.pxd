cdef extern from "vw_py.h":
    void initialize(char*)
    void finish()
    float learn(char*)
