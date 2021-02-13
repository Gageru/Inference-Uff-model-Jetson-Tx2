#include "minorFunctions.h"
#include "tensorNet.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace nvuffparser;
using namespace minorapi;
using namespace cv;
using namespace std;

#define UNLOAD_MODEL 0

int N = 1,
	INPUT_C = 3,
	INPUT_H = 300,
	INPUT_W = 300;
				

char* OUTPUT1 = "mbox_conf_softmax";
char* OUTPUT2 = "";
char* OUTPUT3 = "";
const char* INPUT_BLOB_NAME = "Input";
const char* OUTPUT_BLOB_NAME = "NMS";
	
int OUTPUT_CLS_SIZE = 91;

DetectionOutputParameters detectionOutputParam{true, false,
											   0, OUTPUT_CLS_SIZE,
											   100, 100, 0.5, 0.6,
											   CodeTypeSSD::TF_CENTER, 
											   {0, 2, 1}, true, true};
												   
int main(int argc, char** argv)
{
	TensorNet network(N,
					  INPUT_C,
					  INPUT_H,
					  INPUT_W,
					  OUTPUT_CLS_SIZE,
					  INPUT_BLOB_NAME,
					  OUTPUT_BLOB_NAME);
 
	#if UNLOAD_MODEL == 1
		const char* uffmodel = "/home/jetson-tx2/buildOpenCVTX2/Examples/tensorflow-coco/ssd_inception_v2_coco_2017_11_17/sample_ssd_relu6.uff";
		const char* engine_filepath = "/home/jetson-tx2/buildOpenCVTX2/Examples/tensorflow-coco/ssd_inception_v2_coco_2017_11_17/sample_ssd_relu6.engine";
		
		network.uffToTRTModel(uffmodel);
		network.SaveEngine(engine_filepath);
	#else
		vector<string> classes = {};
		string label_map = "/home/jetson-tx2/buildOpenCVTX2/Examples/tensorflow-coco/ssd_coco_labels.txt";
		string engine_filepath = "/home/jetson-tx2/buildOpenCVTX2/Examples/tensorflow-coco/ssd_inception_v2_coco_2017_11_17/sample_ssd_relu6.engine";
		
		getClasses(label_map, classes);
			
		network.LoadEngine(engine_filepath); 
		
		DimsCHW dimsData = network.getTensorDims(INPUT_BLOB_NAME);
		DimsCHW dimsOut  = network.getTensorDims(OUTPUT_BLOB_NAME);

		cout << "INPUT Tensor Shape is: C: "  <<dimsData.c()<< "  H: "<<dimsData.h()<<"  W:  "<<dimsData.w()<< endl;
		cout << "OUTPUT Tensor Shape is: C: "<<dimsOut.c()<<"  H: "<< dimsOut.h()<<"  W: "<<dimsOut.w()<< endl;

		float* data    = network.allocateMemory( dimsData , (char*)"Input");
		float* output  = network.allocateMemory( dimsOut  , (char*)"NMS");
		
		//frame = imread("/home/jetson-tx2/tensorrt/samples/briKuJEOCDc.jpg" , IMREAD_COLOR);
		Mat frame;
		string pipeline = get_tegra_pipeline(1920, 1080, 30);
		VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
		
		const uint32_t imgWidth  = 300;
		const uint32_t imgHeight = 300;;
		const size_t size = imgWidth * imgHeight * sizeof(float3);
		
		Timer timer;
		timer.startTime();
		
		while(1)
		{
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
			
			network.doInference(&detectionOutputParam, &data[0], &detectionOut[0], &keepCount[0], N);
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
						drawPred(classes, classIndex, confidence, x1, y1, x2, y2, frame_post);
					}	
				}
			}
			putText(frame_post,to_string(timer.getFPS()) + " FPS", Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
			imshow("mobileNet",frame_post);
			
			timer.stopTime();
			
			waitKey(1);
		}
	#endif
    return 0;
}
