#pragma once

#include <opencv2/opencv.hpp>

// 可视化工具类
class VisualizationUtils {
public:
    // 将标记矩阵转换为彩色图像（每个连通域用不同颜色显示）
    // 参数：
    //   labels - 标记矩阵
    //   numComponents - 连通域数量
    //   colorScheme - 配色方案（0=默认，1=替代方案，2=随机）
    static cv::Mat labelsToColorImage(const cv::Mat& labels, int numComponents, int colorScheme = 0);
    
    // 创建叠加图像（将彩色连通域半透明叠加到原图）
    // 参数：
    //   gray - 原始灰度图
    //   labels - 标记矩阵
    //   numComponents - 连通域数量
    //   alpha - 透明度 (0.0-1.0)，越大颜色越明显
    //   colorScheme - 配色方案
    static cv::Mat createOverlay(const cv::Mat& gray, const cv::Mat& labels, 
                                 int numComponents, double alpha = 0.6, int colorScheme = 0);

private:
    // 根据索引和配色方案生成颜色
    static cv::Vec3b generateColor(int index, int colorScheme);
};
