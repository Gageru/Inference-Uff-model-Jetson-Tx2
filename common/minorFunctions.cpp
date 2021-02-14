#include "minorFunctions.h"

namespace minorapi
{	
//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//		
//! Provides the correct flags to initialize the Jetson 
//! camera csi: configures width, height, and frame rate.	
//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//	
	std::string get_tegra_pipeline(int width, int height, int fps) {
	   return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
			  std::to_string(height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(fps) +
			  "/1 ! nvvidconv ! video/x-raw, flip-method=2, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

	}

//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//	
//! Returns the set of objects to be detected. 
//! The list of objects in the labelmap file 
//! looks like this: cat
//!					 dog
//!					 bus 
//!	 				 car
//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//	
	void getClasses(std::string label_map, std::vector<std::string>& classes) {

		std::ifstream file(label_map, std::ios::in);
		
		char buf[256], cuf[256];
		std::string name = "";
		std::string text;
		int begin_pos;
		int end_pos;

		if (!file)
		{
			std::cerr << "File could not be opened" << std::endl;
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
		std::cout << "Label map :" << std::endl;
		for (int i = 0; i < classes.size(); i++) {
			std::cout << std::to_string(i+1) + ") " << classes[i] << std::endl;
		}
	}

//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//	
//! Draws all necessary objects on the frame
//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//	
	void drawPred(std::vector<std::string>& classes, float classId, float conf, 
													 int left, int top, 
													 int right, int bottom, 
													 cv::Mat& frame)
	{
		cv::rectangle(frame, cv::Point(left, top - 13), cv::Point(right, bottom), cv::Scalar(0, 255, 0));

		std::string label = cv::format("%.2f %s", conf, classes[int(classId - 1)].c_str());
		int baseLine;
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		top = cv::max(top, labelSize.height);
		cv::rectangle(frame, cv::Point(left, top - labelSize.height),
							 cv::Point(left + labelSize.width, top + baseLine), 
							 cv::Scalar::all(255), cv::FILLED);
		cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
	}
	
//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//
//! Converts Mat object to vector<float> with subtracting 
//! the average value over the RGB channels of the captured frame	
//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//
	bool loadImageBGR(cv::Mat frame, std::vector<float> &data, const float3& mean )
	{
		const uint32_t imgWidth  = 300;
		const uint32_t imgHeight = 300;
		const uint32_t imgPixels = imgWidth * imgHeight;
		
		for( uint32_t y=0; y < imgHeight; y++ )
		{
			for( uint32_t x=0; x < imgWidth; x++ )
			{
				const float mul = 2.0 / 255.0;
				cv::Vec3b intensity = frame.at<cv::Vec3b>(y,x);
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
//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//	
//! FPS calculation - everything is simple 
//! ... not sure which is correct
//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!//	
	Timer::Timer()
		:fps(0), count(0), tick(0) 
	{
	}
	
	void Timer::startTime()
	{
		timeBegin = time(0);
	}
	
	void Timer::stopTime()
	{
		count++;
		timeNow = time(0) - timeBegin;
		if (timeNow - tick >= 1)
		{
			tick++;
			fps = count;
			count = 0;
		}
	}
	
	int Timer::getFPS()
	{
		return fps;
	}
}



