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
