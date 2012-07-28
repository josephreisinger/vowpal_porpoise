cdef extern from "vw_c.h":
    void initialize(char*)
    void finish()
    float learn(char*)
