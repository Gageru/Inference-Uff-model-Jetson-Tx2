#ifndef TRT_NETWORK_H
#define TRT_NETWORK_H

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvUffParser.h"

#include "buffers.h"
#include "common.h"
#include "logger.h"
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <string>

#include "tensorNet.h"

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

class TensorNet
{
public:
	
	TensorNet(int n,
			  int input_c,
			  int input_h,
			  int input_w,
			  int output_clz_size,
			  const char* input_blob_name,
			  const char* output_blob_name);
	
	~TensorNet();
	
	float* allocateMemory(nvinfer1::DimsCHW dims, char* info);
	
	void uffToTRTModel ( const char* uffmodel);
	
	DimsCHW getTensorDims(const char* name);
	
	inline bool cudaAllocMapped( void** cpuPtr, void** gpuPtr, size_t size);
	
	bool SaveEngine(const std::string& engine_filepath);
    
    bool LoadEngine(const std::string& engine_filepath);
    
    std::vector<std::pair<int64_t, nvinfer1::DataType>>
	calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize);
	
	void doInference(DetectionOutputParameters* detOutParam, float* inputData, float* detectionOut, int* keepCount, int batchSize);

private:

	Logger gLogger;
    Profiler gProfiler;
    
    IRuntime* infer;
    ICudaEngine* engine;
	IExecutionContext *context;
	IHostMemory *gieModelStream;

    const int N,
			  INPUT_C,
			  INPUT_H,
			  INPUT_W,
			  OUTPUT_CLS_SIZE;
			  
	const char* INPUT_BLOB_NAME;
	const char*	OUTPUT_BLOB_NAME;
};
#endif
