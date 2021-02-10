#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <map>
#include <string>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvUffParser.h"
#include <cstring>

#include "BatchStream.h"
#include "NvInferPlugin.h"
#include "EntropyCalibrator.h"

#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "sampleDevice.h"
#include "sampleOptions.h"
#include "sampleEngines.h"
#include "sampleInference.h"
#include "sampleReporting.h"
#include <ctime>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace nvuffparser;
using namespace cv;

static const uint32_t BATCH_SIZE = 1;
std::stringstream gieModelStream;

bool mEnableFP16=false;
bool mOverride16=false;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT1 = "mbox_conf_softmax";
const char* OUTPUT2 = "";
const char* OUTPUT3 = "";
const char* OUTPUT_BLOB_NAME = "detection_out";

class Logger : public ILogger
{
	void log(Severity severity, const char* msg) override
	{
			std::cout << msg << std::endl;
	}
}gLogger;

 struct Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end()) mProfile.push_back(std::make_pair(layerName, ms));
        else record->second += ms;
    }

    void printLayerTimes(const int TIMING_ITERATIONS)
    {

        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
    }

}gProfiler;

 

inline bool cudaAllocMapped( void** cpuPtr, void** gpuPtr, size_t size )
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



bool loadImageBGR( cv::Mat frame, float3** cpu, float3** gpu, int* width, int* height, const float3& mean )
{
	const uint32_t imgWidth  = 300;
	const uint32_t imgHeight = 300;
	const uint32_t imgPixels = imgWidth * imgHeight;
	const size_t   imgSize   = imgWidth * imgHeight * sizeof(float) * 3;

	// allocate buffer for the image

	//if( !cudaAllocMapped((void**)cpu, (void**)gpu, imgSize) )
	//{
	//printf(LOG_CUDA "failed to allocated %zu bytes for image %s\n");
	//	printf("failed to allocated bytes for image");
	//	return false;
	//}
	printf("[cuda] CPU %p GPU %p\n", *cpu, *gpu);
	//float* cpuPtr = (float*)*cpu;

	for( uint32_t y=0; y < imgHeight; y++ )
	{
		for( uint32_t x=0; x < imgWidth; x++ )
		{
			const float mul = 255.0 / 255.0;
			int cn = frame.channels();
            cv::Vec3b intensity = frame.at<cv::Vec3b>(y,x);
			//cout << (float)intensity.val[0]  <<" , "<< (float)intensity.val[1]<<" , "<< (float)intensity.val[2]<<endl;
			const float3 px = make_float3(((float)intensity.val[0] - 104.0f) * mul,
										  ((float)intensity.val[1] - 177.0f) * mul,
										  ((float)intensity.val[2] - 123.0f) * mul );

            // imgPixels * 0 + y * imgWidth + x
			// note:  caffe/GIE is band-sequential (as opposed to the typical Band Interleaved by Pixel)
			cpu[imgPixels * 0 + y * imgWidth + x] = px.x;
			cpu[imgPixels * 1 + y * imgWidth + x] = px.y;
			cpu[imgPixels * 2 + y * imgWidth + x] = px.z;
		}
	}
	return true;

}

DimsCHW getTensorDims(ICudaEngine* engine ,const char* name)
{
    for (int b = 0; b < engine->getNbBindings(); b++) {
        if( !strcmp( name, engine->getBindingName(b)) )
            return static_cast<DimsCHW&&>(engine->getBindingDimensions(b));

    }
    return DimsCHW{0,0,0};

}

float* allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged( &ptr, size*sizeof(float)));

    return ptr;
}



void SaveEngine(const nvinfer1::IHostMemory& trtModelStream, const std::string& engine_filepath)

{
    std::ofstream file;

    file.open(engine_filepath, std::ios::binary | std::ios::out);

    if(!file.is_open())
    {
        std::cout << "read create engine file" << engine_filepath <<" failed" << std::endl;
        return;
    }
    file.write((const char*)trtModelStream.data(), trtModelStream.size());
    file.close();

};

ICudaEngine* LoadEngine(IRuntime& runtime, const std::string& engine_filepath)
{
    std::ifstream file;
    file.open(engine_filepath, std::ios::binary | std::ios::in);
    file.seekg(0, std::ios::end); 
    int length = file.tellg();         
    file.seekg(0, std::ios::beg); 

    std::shared_ptr<char> data(new char[length], std::default_delete<char[]>());
    file.read(data.get(), length);
    file.close();

	initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

    ICudaEngine* engine = runtime.deserializeCudaEngine(data.get(), length, nullptr);
    assert(engine != nullptr);
    
    for (int bi = 0; bi < engine->getNbBindings(); bi++) {
        if (engine->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
        else printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
    }
    return engine;

}

bool caffeToTRTModel (const char* prototxt,
					  const char* caffemodel,
					  std::string engine_filepath)
{
	IHostMemory *serialized{nullptr};

	IHostMemory *deserialized{nullptr};

	IBuilder* builder = createInferBuilder(gLogger);
	initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

	mEnableFP16 = (mOverride16 == true) ? false : builder->platformHasFastFp16();
	printf( "platform %s FP16 support.\n", mEnableFP16 ? "has" : "does not have");
	printf( "loading %s %s\n", prototxt, prototxt);

	nvinfer1::DataType modelDataType = mEnableFP16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;

	
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser *parser = createCaffeParser();

	const IBlobNameToTensor *blobNameToTensor =
				parser->parse(prototxt,		
				caffemodel,		
				*network,		
				modelDataType);

	assert(blobNameToTensor != NULL);

	
	network->markOutput(*blobNameToTensor->find(OUTPUT_BLOB_NAME));
	builder->setMaxBatchSize(1);

	
	builder->setMaxWorkspaceSize(1 << 30);
	
	if(mEnableFP16)

	builder->setHalf2Mode(true);

	builder->setMinFindIterations(10);

	builder->setAverageFindIterations(10); 
	printf("fine till here \n");

	ICudaEngine *engine = builder->buildCudaEngine(*network);
    assert(engine);

    gieModelStream.seekg(0, gieModelStream.beg);

	network->destroy();
 	parser->destroy();

	std::cout << "serializing" << std::endl;

    nvinfer1::IHostMemory* serMem = engine->serialize();

	if( !serMem )
	{
		printf("failed to serialize CUDA engine\n");
		return false;
	}

	gieModelStream.write((const char*)serMem->data(), serMem->size());
	std::cout << "printing size of bytes allocated \t" << (serMem->size())<< std::endl;

	SaveEngine(*serMem, engine_filepath);
	engine->destroy();
    builder->destroy();
    
	return true;
}

void imageInference(ICudaEngine* engine ,void** buffers, int nbBuffer, int batchSize)
{
    std::cout << "Came into the image inference method here. "<< std::endl;
    assert( engine->getNbBindings()==nbBuffer);
    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);
    context->execute(batchSize, buffers);
    context->destroy();

    printf("[OTLDK] -- Step2 --\n");

}

bool uffToTRTModel ( const char* uffmodel, 
					 const char* engine_filepath)
{

	IBuilder* builder = createInferBuilder(gLogger);
	INetworkDefinition* network = builder->createNetwork();
	
	initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
	IUffParser* parser = createUffParser();
	
	parser->registerInput("Input", DimsCHW(3, 300, 300), UffInputOrder::kNCHW);
	parser->registerOutput("MarkOutput_0");
	
	parser->parse(uffmodel, *network, nvinfer1::DataType::kFLOAT);
	
	builder->setMaxBatchSize(1);
	builder->setMaxWorkspaceSize(1 << 30);
	ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    gieModelStream.seekg(0, gieModelStream.beg);

	network->destroy();
 	parser->destroy();

	std::cout << "serializing" << std::endl;

    nvinfer1::IHostMemory* serMem = engine->serialize();

	if( !serMem )
	{
		printf("failed to serialize CUDA engine\n");
		return false;
	}

	gieModelStream.write((const char*)serMem->data(), serMem->size());
	std::cout << "printing size of bytes allocated \t" << (serMem->size())<< std::endl;

	SaveEngine(*serMem, engine_filepath);
	
	engine->destroy();
    builder->destroy();
    
	return true;
}

std::string get_tegra_pipeline(int width, int height, int fps) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
           std::to_string(height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(fps) +
           "/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

}

int main(int argc, char** argv)
{

	//const char* prototxt = "/home/jetson-tx2/buildOpenCVTX2/Examples/VOC0712_SSD/deploy_ssd.prototxt";
	//const char* caffemodel = "/home/jetson-tx2/buildOpenCVTX2/Examples/VOC0712_SSD/VGG_VOC0712_SSD_300x300_iter_120000_caffemodel.caffemodel";
	//std::string engine_filepath = "/home/jetson-tx2/buildOpenCVTX2/Examples/VOC0712_SSD/VOC0712.engine";

	//const char* prototxt = "/home/jetson-tx2/buildOpenCVTX2/Examples/deploy_up.prototxt";
	//const char* caffemodel = "/home/jetson-tx2/buildOpenCVTX2/Examples/VGG_coco_SSD_300x300_iter_400000.caffemodel";
	//std::string engine_filepath = "/home/jetson-tx2/buildOpenCVTX2/Examples/COCO81.engine";

	//std::cout << "Starting execution" << std::endl;

	/*if (!caffeToTRTModel(prototxt, caffemodel, engine_filepath))
	{
		printf("error in model serialization\n");
		return 0;
	}*/
	
	/*const char* uffmodel = "/home/jetson-tx2/buildOpenCVTX2/Examples/tensorflow-coco/ssd_inception_v2_coco_2017_11_17/sample_ssd_relu6.uff";
	const char* engine_filepath = "/home/jetson-tx2/buildOpenCVTX2/Examples/tensorflow-coco/ssd_inception_v2_coco_2017_11_17/sample_ssd_relu6.engine";
	if (!uffToTRTModel(uffmodel, engine_filepath))
	{
		printf("error in model serialization\n");
		return 0;

	}*/

	//std::string pipeline = get_tegra_pipeline(640, 480, 30);
	//VideoCapture cap(pipeline, CAP_GSTREAMER);

    std::string engine_filepath = "/home/jetson-tx2/buildOpenCVTX2/Examples/VOC0712_SSD/VOC0712.engine";
    nvinfer1::IRuntime* infer = createInferRuntime(gLogger);  

    ICudaEngine* engine = LoadEngine(*infer, engine_filepath); 

	//create context for execution
	IExecutionContext *context = engine->createExecutionContext();

    DimsCHW dimsData = getTensorDims(engine, INPUT_BLOB_NAME);
    DimsCHW dimsOut  = getTensorDims(engine, OUTPUT_BLOB_NAME);
    //DimsCHW dims1    = getTensorDims(engine, OUTPUT1);

    std::cout << "INPUT Tensor Shape is: C: "  <<dimsData.c()<< "  H: "<<dimsData.h()<<"  W:  "<<dimsData.w()<< std::endl;
    std::cout << "OUTPUT Tensor Shape is: C: "<<dimsOut.c()<<"  H: "<< dimsOut.h()<<"  W: "<<dimsOut.w()<<std::endl;

    float* data    = allocateMemory( dimsData , (char*)"input");
    float* output  = allocateMemory( dimsOut  , (char*)"detection_out");
    //float* output1 = allocateMemory( dims1    , (char*)"mbox_conf_softmax");

    int height = 300;
    int width  = 300;

	Mat frame;
    Mat frame_float;
    
    frame = cv::imread("/home/jetson-tx2/tensorrt/samples/aNLAsSNKrzQ.jpg" , cv::IMREAD_COLOR);
    resize(frame, frame, Size(300,300));
	time_t timeBegin = time(0);
	time_t timeNow;
	int fps;
	int count = 0;
	int tick = 0;
    //void* imgCPU;
	

    while(1)
    {
	    void* imgCUDA;
		count++;
		//cap.read(frame);
		resize(frame, frame, Size(300,300));
		const size_t size = width * height * sizeof(float3);
		
		if( cudaMalloc( &imgCUDA, size) )
		{
			std::cout <<"Cuda Memory allocation error occured."<< std::endl;
			return 0;
		}
		void* imgDATA = malloc(size);
		memset(imgDATA,0,size);
		if( !loadImageBGR(frame, (float3**)&imgDATA, (float3**)&imgCUDA, &height, &width, make_float3(104.0f,177.0f,123.0f)))
        {
            printf("failed to load image '%s'\n", "Image");
            return 0;
        }
		cudaMemcpyAsync(imgCUDA,imgDATA,size,cudaMemcpyHostToDevice);
		void* buffers[] = {imgCUDA, output};
		printf("[OTLDK] -- Step1 --\n");

		imageInference(engine, buffers, 2, BATCH_SIZE);
		
		std::vector<std::vector<float> > detections;
		
		for (int k=0; k<3; k++)
		{
			if(output[7*k + 1] == -1)
				break;
			float classIndex = output[7*k + 1];
			float confidence = output[7*k + 2];
			float xmin = output[7*k + 3];
			float ymin = output[7*k + 4];
			float xmax = output[7*k + 5];
			float ymax = output[7*k + 6];
			int x1 = static_cast<int>(xmin * frame.cols);
			int y1 = static_cast<int>(ymin * frame.rows);
			int x2 = static_cast<int>(xmax * frame.cols);
			int y2 = static_cast<int>(ymax * frame.rows);
			std::cout << classIndex << "; " << confidence << "; "  << x1 << "; " << y1 << "; " << x2<< "; " << y2 << std::endl;
			cv::rectangle(frame,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(255,0,255),1);
			std::string label = format("%.2f %.f", confidence, classIndex);
			cv::putText(frame, label, cv::Point(x1,y2),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0));
		}
		time_t timeNow = time(0) - timeBegin;
		if (timeNow - tick >= 1)
		{
			tick++;
			fps = count;
			count = 0;
		}
		putText(frame, std::to_string(fps) + " FPS", Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		imshow("mobileNet",frame);
		free(imgDATA);
		cudaFree(imgCUDA);
		//cudaFree(imgCPU);
		waitKey(0);

    }

   // cudaFree(imgCUDA);
   // cudaFreeHost(imgCPU);
    cudaFree(output);
    engine->destroy();
    infer->destroy();
    return 0;
}


