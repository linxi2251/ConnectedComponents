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
        double meanIoU; // 与基准算法的平均IoU（组件级）
        // 原始连通域标签（CV_32S），用于评估
        cv::Mat labels;
        cv::Mat color;
    };

    static std::vector<Result> evaluate(const cv::Mat& binary, const std::vector<IComponentDetector*>& detectors);
};
