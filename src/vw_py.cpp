#include "vw_py.h"

#include <iostream>
#include <vowpalwabbit/vw.h>
#include <vowpalwabbit/example.h>

vw* _vw = NULL;

void initialize(char* params) {
    _vw = &VW::initialize(std::string(params));
}

void finalize() {
    if (_vw != NULL) {
        VW::finish(*_vw);
    }
}

float learn(char* line) {
    if (_vw == NULL) {
        cout << "attemping to learn before initialization";
    }
    example* v = VW::read_example(*_vw, line);
    _vw->learn(_vw, v);
    float p = v->final_prediction;
    VW::finish_example(*_vw, v);
    return p;
}
