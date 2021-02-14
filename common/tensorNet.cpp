#include "tensorNet.h"

void Logger::log(Severity severity, const char* msg)
{
		std::cout << msg << std::endl;
}

void Profiler::reportLayerTime(const char* layerName, float ms)
{
	 auto record = find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
	 if (record == mProfile.end()) mProfile.push_back(std::make_pair(layerName, ms));
     else record->second += ms;
}

void Profiler::printLayerTimes(const int TIMING_ITERATIONS)
{
	float totalTime = 0;
	for (size_t i = 0; i < mProfile.size(); i++)
	{
		printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
		totalTime += mProfile[i].second;
	}
	printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
}

//!~~~~~~~~~~~~~!//	
//!	Maine class 
//!~~~~~~~~~~~~~!//	

//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//
//! Initiating an object with input parameters:
//!	n - batch size, input_c - number of color channels of the frame,
//!	input_h - frame height, input_w - frame width
//! input_blob_name - the name of the input network layer
//! output_blob_name - the name of the output network layer
//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//
TensorNet::TensorNet(int n,
					 int input_c,
					 int input_h,
					 int input_w,
					 int output_clz_size,
					 const char* input_blob_name,
					 const char* output_blob_name)
	:N(n), 
	 INPUT_C(input_c), 
	 INPUT_H(input_h), 
	 INPUT_W(input_w), 
	 OUTPUT_CLS_SIZE(output_clz_size), 
	 INPUT_BLOB_NAME(input_blob_name), 
	 OUTPUT_BLOB_NAME(output_blob_name)
{
    IHostMemory *gieModelStream = nullptr;
    IRuntime* infer = nullptr;
    ICudaEngine* engine = nullptr;
	IExecutionContext *context = nullptr;
}

//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//
//! We destroy all the periphery of the engine
//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//
TensorNet::~TensorNet()
{
	std::cout << "Destroy network:\n";
	
	std::cout << "gieModelStream->destroy()\n";
	if(!gieModelStream)
		gieModelStream->destroy();
	
    std::cout << "infer->destroy()\n";
    if(!infer)
		infer->destroy();
    
    std::cout << "engine->destroy()\n";
    if(!engine)
		engine->destroy();
    
    std::cout << "context->destroy();\n";
    if(!context)
		context->destroy();
}

//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//
//! Allocates memory that will be automatically 
//! managed by the Unified Memory system.
//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//
float* TensorNet::allocateMemory(nvinfer1::DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    size = N * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged( &ptr, size*sizeof(float)));

    return ptr;
}

//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//
//! Converts Uff model to optimized engine
//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//
void TensorNet::uffToTRTModel ( const char* uffmodel)
{

	IBuilder* builder = createInferBuilder(gLogger);
	INetworkDefinition* net = builder->createNetwork();
	
	initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
	nvuffparser::IUffParser* parser = nvuffparser::createUffParser();
	
	parser->registerInput("Input", DimsCHW(3, 300, 300), nvuffparser::UffInputOrder::kNCHW);
	parser->registerOutput("MarkOutput_0");
	
	parser->parse(uffmodel, *net, nvinfer1::DataType::kFLOAT);
	
	builder->setMaxBatchSize(1);
	builder->setMaxWorkspaceSize(1 << 26);

	builder->setFp16Mode(builder->platformHasFastFp16());
	
	this->engine = builder->buildCudaEngine(*net);
    assert(engine);

	net->destroy();
 	parser->destroy();
	builder->destroy();
}

//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//
//! Linking to input and output layers
//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//
DimsCHW TensorNet::getTensorDims(const char* name)
{
    for (int b = 0; b < engine->getNbBindings(); b++) {
        if( !strcmp( name, engine->getBindingName(b)) )
            return static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
    }
    return DimsCHW{0,0,0};
}

//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//
//! Allocates the required 
//! amount of memory for the host
//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//
inline bool TensorNet::cudaAllocMapped( void** cpuPtr, void** gpuPtr, size_t size )
{
	if( !cpuPtr || !gpuPtr || size == 0 )
		return false;

	//CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));

	if( cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped) )
		return false;
		
	if( cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0) )
		return false;
		
	memset(*cpuPtr, 0, size);
	printf("[cuda] cudaAllocMapped %zu bytes, CPU %p GPU %p\n", size, *cpuPtr, *gpuPtr);
	return true;

}

bool TensorNet::SaveEngine(const std::string& engine_filepath)
{
	std::cout << "~~~~~~~~Serializing model~~~~~~~~" << std::endl;

    this->gieModelStream = this->engine->serialize();

	if( !gieModelStream )
	{
		printf("failed to serialize CUDA engine\n");
		return false;
	}
	
    std::ofstream file;

    file.open(engine_filepath, std::ios::binary | std::ios::out);

    if(!file.is_open())
    {
        std::cout << "read create engine file" << engine_filepath <<" failed" << std::endl;
        return false;
    }
    std::cout << "Printing size of bytes allocated [" << gieModelStream->size()<< ']' << std::endl;
    
    file.write((const char*)gieModelStream->data(), gieModelStream->size());
    file.close();
    
    this->gieModelStream->destroy();
    
    return 1;
}

bool TensorNet::LoadEngine(const std::string& engine_filepath)
{
	
	this->infer = createInferRuntime(gLogger);
	
    std::ifstream file;
    file.open(engine_filepath, std::ios::binary | std::ios::in);
    file.seekg(0, std::ios::end); 
    int length = file.tellg();         
    file.seekg(0, std::ios::beg); 

    std::shared_ptr<char> data(new char[length], std::default_delete<char[]>());
    file.read(data.get(), length);
    file.close();

	initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

    this->engine = this->infer->deserializeCudaEngine(data.get(), length, nullptr);
    assert(engine != nullptr);
    
    for (int bi = 0; bi < engine->getNbBindings(); bi++) {
        if (engine->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
        else printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
    }
    
    this->context = this->engine->createExecutionContext();
	assert(context != nullptr);
	
    return 1;
}

std::vector<std::pair<int64_t, nvinfer1::DataType>>
TensorNet::calculateBindingBufferSizes(const nvinfer1::ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, nvinfer1::DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        nvinfer1::DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = samplesCommon::volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

void TensorNet::doInference(DetectionOutputParameters* detOutParam ,float* inputData, float* detectionOut, int* keepCount, int batchSize)
{				
    //const ICudaEngine& engine = this->context->getEngine();
    // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 2 output.
    int nbBindings = engine->getNbBindings();

    std::vector<void*> buffers(nbBindings);
    std::vector<std::pair<int64_t, nvinfer1::DataType>> buffersSizes = calculateBindingBufferSizes(*engine, nbBindings, batchSize);

    for (int i = 0; i < nbBindings; ++i)
    {
        auto bufferSizesOutput = buffersSizes[i];
        buffers[i] = samplesCommon::safeCudaMalloc(bufferSizesOutput.first * samplesCommon::getElementSize(bufferSizesOutput.second));
    }

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings().
    int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME),
        outputIndex0 = engine->getBindingIndex(OUTPUT_BLOB_NAME),
        outputIndex1 = outputIndex0 + 1; //engine.getBindingIndex(OUTPUT_BLOB_NAME1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream);

    auto t_start = std::chrono::high_resolution_clock::now();
    context->execute(batchSize, &buffers[0]);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    std::cout << "Time taken for inference is " << total << " ms." << std::endl;

    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
    {
        if (engine->bindingIsInput(bindingIdx))
            continue;
        auto bufferSizesOutput = buffersSizes[bindingIdx];
    }

    cudaMemcpyAsync(detectionOut, buffers[outputIndex0], batchSize * detOutParam->keepTopK * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(keepCount, buffers[outputIndex1], batchSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex0]);
    cudaFree(buffers[outputIndex1]);
}
