#ifndef VW_SINGLETON_H
#define VW_SINGLETON_H

#include <vowpalwabbit/vw.h>
#include <vowpalwabbit/example.h>

class vw_py {
    public:
        vw_py(char* params) {
            _vw = VW::initialize(std::string(params));
        }
        ~vw_py() {
            VW::finish(_vw);
        }

        simple_prediction learn(char*);

    private:
        // private copy constructor
        vw_py(vw_py const&){}; 

        vw _vw;
}

#endif
