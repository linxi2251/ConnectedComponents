#pragma once

#include "IComponentDetector.h"
#include <vector>

class ComponentEvaluator {
public:
    struct Result {
        std::string name;
        int numComponents;
        double timeMs;
        double pixelAccuracy;
        double meanIoU;
        cv::Mat labels;
        cv::Mat color;
    };

    static std::vector<Result> evaluate(const cv::Mat& binary, const std::vector<IComponentDetector*>& detectors);
};
