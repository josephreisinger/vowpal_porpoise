#include "vw_c.h"

#include <iostream>
#include <vowpalwabbit/vw.h>
#include <vowpalwabbit/example.h>
#include <cassert>

vw _vw;
bool _initialized = false;

void initialize(char* params) {
    assert(!_initialized);
    cout << "initialized with params=[" << std::string(params) << "]" << endl;
    _vw = VW::initialize(std::string(params));
    _initialized = true;
}

void finish() {
    assert(_initialized);
    VW::finish(_vw);
}

float learn(char* line) {
    assert(_initialized);

    example* v = VW::read_example(_vw, line);
    _vw.learn(&_vw, v);
    float p = v->final_prediction;
    VW::finish_example(_vw, v);

    return p;
}
