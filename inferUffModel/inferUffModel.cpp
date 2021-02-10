#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <string>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvUffParser.h"

#include "BatchStream.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace nvuffparser;
using namespace cv;
using namespace std;

static const uint32_t BATCH_SIZE = 1;
stringstream gieModelStream;

bool mEnableFP16=false;
bool mOverride16=false;

const char* INPUT_BLOB_NAME = "Input";
const char* OUTPUT1 = "mbox_conf_softmax";
const char* OUTPUT2 = "";
const char* OUTPUT3 = "";
const char* OUTPUT_BLOB_NAME = "NMS";

const int 	INPUT_C = 3,
			INPUT_H = 300,
			INPUT_W = 300,
			N =1;
			  
static constexpr int OUTPUT_CLS_SIZE = 91;

DetectionOutputParameters detectionOutputParam{true, false, 0, OUTPUT_CLS_SIZE, 100, 100, 0.5, 0.6, CodeTypeSSD::TF_CENTER, {0, 2, 1}, true, true};

class Logger : public ILogger
{
	void log(Severity severity, const char* msg) override
	{
			cout << msg << endl;
	}
}gLogger;

 struct Profiler : public IProfiler
{
    typedef pair<string, float> Record;
    vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end()) mProfile.push_back(make_pair(layerName, ms));
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

bool loadImageBGR( Mat frame, vector<float> &data, const float3& mean )
{
	const uint32_t imgWidth  = 300;
	const uint32_t imgHeight = 300;
	const uint32_t imgPixels = imgWidth * imgHeight;
	
	for( uint32_t y=0; y < imgHeight; y++ )
	{
		for( uint32_t x=0; x < imgWidth; x++ )
		{
			const float mul = 2.0 / 255.0;
            Vec3b intensity = frame.at<Vec3b>(y,x);
			//cout << (float)intensity.val[0]  <<" , "<< (float)intensity.val[1]<<" , "<< (float)intensity.val[2]<<endl;
			const float3 px = make_float3(((float)intensity.val[0] - mean.x) * mul,
										  ((float)intensity.val[1] - mean.y) * mul,
										  ((float)intensity.val[2] - mean.z) * mul );

            // imgPixels * 0 + y * imgWidth + x
			// note:  caffe/GIE is band-sequential (as opposed to the typical Band Interleaved by Pixel)
			data[imgPixels * 0 + y * imgWidth + x] = px.x;
			data[imgPixels * 1 + y * imgWidth + x] = px.y;
			data[imgPixels * 2 + y * imgWidth + x] = px.z;
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
    cout << "Allocate memory: " << info << endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged( &ptr, size*sizeof(float)));

    return ptr;
}

void SaveEngine(const nvinfer1::IHostMemory& trtModelStream, const string& engine_filepath)
{
    ofstream file;

    file.open(engine_filepath, ios::binary | ios::out);

    if(!file.is_open())
    {
        cout << "read create engine file" << engine_filepath <<" failed" << endl;
        return;
    }
    file.write((const char*)trtModelStream.data(), trtModelStream.size());
    file.close();

};

ICudaEngine* LoadEngine(IRuntime& runtime, const string& engine_filepath)
{
    ifstream file;
    file.open(engine_filepath, ios::binary | ios::in);
    file.seekg(0, ios::end); 
    int length = file.tellg();         
    file.seekg(0, ios::beg); 

    shared_ptr<char> data(new char[length], default_delete<char[]>());
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

vector<pair<int64_t, nvinfer1::DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    vector<pair<int64_t, nvinfer1::DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        nvinfer1::DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = samplesCommon::volume(dims) * batchSize;
        sizes.push_back(make_pair(eltCount, dtype));
    }

    return sizes;
}


void doInference(IExecutionContext& context, float* inputData, float* detectionOut, int* keepCount, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 2 output.
    int nbBindings = engine.getNbBindings();

    vector<void*> buffers(nbBindings);
    vector<pair<int64_t, nvinfer1::DataType>> buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    for (int i = 0; i < nbBindings; ++i)
    {
        auto bufferSizesOutput = buffersSizes[i];
        buffers[i] = samplesCommon::safeCudaMalloc(bufferSizesOutput.first * samplesCommon::getElementSize(bufferSizesOutput.second));
    }

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings().
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
        outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME),
        outputIndex1 = outputIndex0 + 1; //engine.getBindingIndex(OUTPUT_BLOB_NAME1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream);

    auto t_start = chrono::high_resolution_clock::now();
    context.execute(batchSize, &buffers[0]);
    auto t_end = chrono::high_resolution_clock::now();
    float total = chrono::duration<float, milli>(t_end - t_start).count();

    cout << "Time taken for inference is " << total << " ms." << endl;

    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
    {
        if (engine.bindingIsInput(bindingIdx))
            continue;
        auto bufferSizesOutput = buffersSizes[bindingIdx];
    }

    cudaMemcpyAsync(detectionOut, buffers[outputIndex0], batchSize * detectionOutputParam.keepTopK * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(keepCount, buffers[outputIndex1], batchSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex0]);
    cudaFree(buffers[outputIndex1]);
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
	builder->setMaxWorkspaceSize(1 << 26);

	builder->setFp16Mode(builder->platformHasFastFp16());
	
	ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    gieModelStream.seekg(0, gieModelStream.beg);

	network->destroy();
 	parser->destroy();

	cout << "serializing" << endl;

    nvinfer1::IHostMemory* serMem = engine->serialize();

	if( !serMem )
	{
		printf("failed to serialize CUDA engine\n");
		return false;
	}

	gieModelStream.write((const char*)serMem->data(), serMem->size());
	cout << "printing size of bytes allocated \t" << (serMem->size())<< endl;

	SaveEngine(*serMem, engine_filepath);
	
	engine->destroy();
    builder->destroy();
    
	return true;
}

string get_tegra_pipeline(int width, int height, int fps) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + to_string(width) + ", height=(int)" +
           to_string(height) + ", format=(string)NV12, framerate=(fraction)" + to_string(fps) +
           "/1 ! nvvidconv ! video/x-raw, flip-method=2, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

}

string locateFile(const string& input)
{
    vector<string> dirs{"data/ssd/",
                                  "data/ssd/VOC2007/",
                                  "/home/jetson-tx2/tensorrt/samples/",
                                  "data/samples/ssd/",
                                  "data/int8_samples/ssd/",
                                  "int8/ssd/",
                                  "data/samples/ssd/VOC2007/",
                                  "data/samples/ssd/VOC2007/PPMImages/"};
    return locateFile(input, dirs);
}

vector<string> classes = {};

void getClasses(string label_map) {

	ifstream file(label_map, ios::in);
	
	char buf[256], cuf[256];
	string name = "";
	string text;
	int begin_pos;
	int end_pos;

	if (!file)
	{
		cerr << "File could not be opened" << endl;
		system("pause");
		exit(1);
	}

	while (!file.eof())
	{
		file.getline(buf, sizeof(buf));
		text = buf;
		classes.push_back(text);
		text.clear();
	}
	cout << "Label map :" << endl;
	for (int i = 0; i < classes.size(); i++) {
		cout << to_string(i+1) + ") " << classes[i] << endl;
	}
}

void drawPred(float classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	rectangle(frame, Point(left, top - 13), Point(right, bottom), Scalar(0, 255, 0));

	std::string label = format("%.2f %s", conf, classes[int(classId - 1)].c_str());
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - labelSize.height),
		Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

int main(int argc, char** argv)
{
	/*const char* uffmodel = "/home/jetson-tx2/buildOpenCVTX2/Examples/tensorflow-coco/ssd_inception_v2_coco_2017_11_17/sample_ssd_relu6.uff";
	const char* engine_filepath = "/home/jetson-tx2/buildOpenCVTX2/Examples/tensorflow-coco/ssd_inception_v2_coco_2017_11_17/sample_ssd_relu6.engine";
	if (!uffToTRTModel(uffmodel, engine_filepath))
	{
		printf("error in model serialization\n");
		return 0;
	}*/
	
	string label_map = "/home/jetson-tx2/buildOpenCVTX2/Examples/tensorflow-coco/ssd_coco_labels.txt";
	getClasses(label_map);
	
    string engine_filepath = "/home/jetson-tx2/buildOpenCVTX2/Examples/tensorflow-coco/ssd_inception_v2_coco_2017_11_17/sample_ssd_relu6.engine";
    nvinfer1::IRuntime* infer = createInferRuntime(gLogger);  

    ICudaEngine* engine = LoadEngine(*infer, engine_filepath); 

	IExecutionContext *context = engine->createExecutionContext();
	assert(context != nullptr);
	
    DimsCHW dimsData = getTensorDims(engine, INPUT_BLOB_NAME);
    DimsCHW dimsOut  = getTensorDims(engine, OUTPUT_BLOB_NAME);

    cout << "INPUT Tensor Shape is: C: "  <<dimsData.c()<< "  H: "<<dimsData.h()<<"  W:  "<<dimsData.w()<< endl;
    cout << "OUTPUT Tensor Shape is: C: "<<dimsOut.c()<<"  H: "<< dimsOut.h()<<"  W: "<<dimsOut.w()<< endl;

    //float* data    = allocateMemory( dimsData , (char*)"Input");
    //float* output  = allocateMemory( dimsOut  , (char*)"NMS");
    
    //frame = imread("/home/jetson-tx2/tensorrt/samples/briKuJEOCDc.jpg" , IMREAD_COLOR);
    Mat frame;
    string pipeline = get_tegra_pipeline(1920, 1080, 30);
	VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
	
    const uint32_t imgWidth  = 300;
	const uint32_t imgHeight = 300;;
	const size_t size = imgWidth * imgHeight * sizeof(float3);
	
	time_t timeBegin = time(0);
	int fps;
	int count = 0;
	int tick = 0;
	
	while(1)
	{
		count++;
		cap.read(frame);
		Mat frame_post = frame.clone();
		resize(frame, frame, Size(300,300));
		//cvtColor(frame, frame, COLOR_BGRA2BGR);
		vector<float> data(size);
		
		if( !loadImageBGR(frame, data, make_float3(104.0f,177.0f,123.0f)))
        {
            printf("failed to load image '%s'\n", "Image");
            return 0;
        }
		
		vector<float> detectionOut(N * detectionOutputParam.keepTopK * 7);
		vector<int> keepCount(N);
		
		doInference(*context, &data[0], &detectionOut[0], &keepCount[0], N);
		cout<<"//--------~---------//"<<endl;
		for (int p = 0; p < N; ++p)
		{
			for (int i = 0; i < keepCount[p]; ++i)
			{
				float* det = &detectionOut[0] + (p * detectionOutputParam.keepTopK + i) * 7;
				if (det[2] >= 0.4f) 
				{
					cout << "Detected " << det[1]
							 << " with confidence " << det[2] * 100.f << " and coordinates ("
							 << det[3] * INPUT_W << "," << det[4] * INPUT_H << ")"
							 << ",(" << det[5] * INPUT_W << "," << det[6] * INPUT_H << ")."<< endl;
					float classIndex = det[1];
					float confidence = det[2];
					float xmin = det[3];
					float ymin = det[4];
					float xmax = det[5];
					float ymax = det[6];
					int x1 = static_cast<int>(xmin * frame.cols) * 1920.0 / 300.0;
					int y1 = static_cast<int>(ymin * frame.rows) * 1080.0 / 300.0;
					int x2 = static_cast<int>(xmax * frame.cols) * 1920.0 / 300.0;
					int y2 = static_cast<int>(ymax * frame.rows) * 1080.0 / 300.0;
					cout << classIndex << "; " << confidence << "; "  << x1 << "; " << y1 << "; " << x2<< "; " << y2 << endl;
					drawPred(classIndex, confidence, x1, y1, x2, y2, frame_post);
				}	
			}
		}
		putText(frame_post,to_string(fps) + " FPS", Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		imshow("mobileNet",frame_post);
		
		time_t timeNow = time(0) - timeBegin;
			if (timeNow - tick >= 1)
			{
				tick++;
				fps = count;
				count = 0;
			}
		waitKey(1);
	}
	
    engine->destroy();
    infer->destroy();
    //
    return 0;
}
