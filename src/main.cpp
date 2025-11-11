#include "ComponentEvaluator.h"
#include "ConnectedComponentsBFS.h"
#include "ConnectedComponentsUF.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>

// 题目：
// 在二值图像中，有多个相互连通的区域，请写一程序将所有的连通域提取出来，并显示出提取的效果（每个连通域用不同颜色显示）
// 要求：
// 1. opencv & c/c++， 可以上网查阅资料，但代码要自己独立实现
// 2. 用自己算法实现，不可以用opencv的轮廓提取、填充功能（因为它不够准确）
int main(int argc, char** argv) {
    std::string imagePath = "../input.jpg";
    if (argc > 1) imagePath = argv[1];

    std::cout << "Trying to read: " << imagePath << std::endl;
    std::cout << "Current path: " << std::filesystem::current_path() << std::endl;

    cv::Mat gray = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return -1;
    }

    // 预处理：二值化
    cv::Mat binary;
    cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY);
    // 形态学闭运算，填充空洞、连接断裂区域
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);

    ConnectedComponentsBFS bfs;
    // bfs.setEightConnectivity(true);
    ConnectedComponentsUF uf;
    // uf.setEightConnectivity(true);

    std::vector<IComponentDetector*> detectors = { &bfs, &uf };
    auto results = ComponentEvaluator::evaluate(binary, detectors);


    std::cout << "评估结果：" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "算法名称\t\t连通域数\t\t耗时(ms)\t\t与基准一致率\tmeanIoU" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    for (auto& r : results)
       std::cout << std::setw(20) << std::left << r.name
           << "\t" << r.numComponents
           << "\t\t" << std::fixed << std::setprecision(2) << r.timeMs
           << "\t\t" << (r.pixelAccuracy*100) << "%"
           << "\t" << std::fixed << std::setprecision(3) << r.meanIoU
           << std::endl;
    // 保存结果
    for (auto& r : results) {
        std::string colorPath = r.name + "_color.png";
        cv::imwrite(colorPath, r.color);
        std::cout << "结果已保存: " << colorPath << std::endl;
    }
    return 0;
}
