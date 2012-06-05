cdef extern from "vw_py.h":
    void initialize(char*)
    void finalize()
    float learn(char*)
