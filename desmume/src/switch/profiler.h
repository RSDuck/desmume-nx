#pragma once

#include <stdint.h>

namespace profiler
{

class Section
{
    const char* name_;
    uint64_t startTick_;

public:
    Section(const char* name);
    ~Section();

};

void frame();

}