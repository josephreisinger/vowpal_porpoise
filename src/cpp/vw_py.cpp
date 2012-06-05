#include "vw_py.h"

simple_prediction vw_py::learn(char* example) {
    example* v = VW::read_example(_vw, example);
    _vw.learn(_vw, v);
    simple_prediction p = v->final_prediction;
    VW::finish_example(_vw, v);
    return p;
}
