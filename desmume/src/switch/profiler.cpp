#include "profiler.h"

#include <map>
#include <stdio.h>

#include <switch.h>

namespace profiler
{

struct ProfileData
{
    uint64_t timeSpend;
    int hit;
};

static std::map<const char*, ProfileData> dataset;
static int frames = 0;

Section::Section(const char* name) : name_(name)
{
    startTick_ = armGetSystemTick();
}
Section::~Section()
{
    dataset[name_].timeSpend += armGetSystemTick()-startTick_;
    dataset[name_].hit++;
}

void frame()
{
    printf("profiling frame %d\n", frames++);
    for (auto data : dataset)
    {
        printf("  %s: hit %dx %fms\n", data.first, data.second.hit, ((float)data.second.timeSpend)/armGetSystemTickFreq()*1000.f);
    }
    dataset.clear();
}

}