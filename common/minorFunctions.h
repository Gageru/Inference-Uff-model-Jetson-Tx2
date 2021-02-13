#ifndef TRT_MINOR_FUNCTIONS_H
#define TRT_MINOR_FUNCTIONS_H

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <cuda_runtime.h>
#include <time.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>

#include "minorFunctions.h"

namespace minorapi
{
	std::string get_tegra_pipeline(int width, int height, int fps);
	
	void getClasses(std::string label_map, std::vector<std::string>& classes);
	
	void drawPred(std::vector<std::string>& classes, float classId, float conf, 
													 int left, int top, 
													 int right, int bottom, 
													 cv::Mat& frame);
	 
	bool loadImageBGR(cv::Mat frame, std::vector<float> &data, const float3& mean);
	
	class Timer
	{
	public:
		Timer();
		
		void startTime();
		void stopTime();
		
		int getFPS();
		
	private:
		int fps;
		int count;
		int tick;
		time_t timeNow;
		time_t timeBegin;
	};
}
#endif
