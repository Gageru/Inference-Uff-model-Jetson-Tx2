#ifndef TRT_NET_H
#define TRT_NET_H

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <cuda_runtime.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>

#include <time.h>
#include <cuda.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvUffParser.h"

#include "buffers.h"
#include "common.h"
#include "logger.h"

class Logger : public ILogger
{
	void log(Severity severity, const char* msg) override;
};

struct Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms);
    
    void printLayerTimes(const int TIMING_ITERATIONS);

};

class TensorNetwork
{
	
};

#endif
