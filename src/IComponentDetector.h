#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class IComponentDetector {
public:
    virtual ~IComponentDetector() = default;

    virtual cv::Mat detect(const cv::Mat& binary, int minSize = 0) = 0;
    virtual std::string name() const = 0;
    virtual int numComponents() const = 0;
};
