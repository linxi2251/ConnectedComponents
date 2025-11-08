#include "VisualizationUtils.h"

cv::Vec3b VisualizationUtils::generateColor(int index, int colorScheme) {
    float hue;
    
    switch (colorScheme) {
        case 1: // 替代配色方案
            hue = ((index * 37) + 90) % 180;
            break;
        case 2: // 随机配色
            std::srand(index * 12345);
            hue = std::rand() % 180;
            break;
        case 0: // 默认配色方案
        default:
            hue = (index * 45) % 180;
            break;
    }
    
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr.at<cv::Vec3b>(0, 0);
}

cv::Mat VisualizationUtils::labelsToColorImage(const cv::Mat& labels, int numComponents, int colorScheme) {
    int rows = labels.rows;
    int cols = labels.cols;
    cv::Mat colorImg(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
    
    for (int i = 1; i <= numComponents; ++i) {
        cv::Vec3b color = generateColor(i, colorScheme);
        
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (labels.at<int>(y, x) == i) {
                    colorImg.at<cv::Vec3b>(y, x) = color;
                }
            }
        }
    }
    
    return colorImg;
}

cv::Mat VisualizationUtils::createOverlay(const cv::Mat& gray, const cv::Mat& labels, 
                                          int numComponents, double alpha, int colorScheme) {
    cv::Mat overlay;
    cv::cvtColor(gray, overlay, cv::COLOR_GRAY2BGR);
    
    // 先生成彩色图像
    cv::Mat colorImg = labelsToColorImage(labels, numComponents, colorScheme);
    
    // alpha混合
    for (int y = 0; y < overlay.rows; ++y) {
        for (int x = 0; x < overlay.cols; ++x) {
            if (labels.at<int>(y, x) > 0) {
                cv::Vec3b& pixel = overlay.at<cv::Vec3b>(y, x);
                cv::Vec3b color = colorImg.at<cv::Vec3b>(y, x);
                pixel[0] = cv::saturate_cast<uchar>(alpha * color[0] + (1 - alpha) * pixel[0]);
                pixel[1] = cv::saturate_cast<uchar>(alpha * color[1] + (1 - alpha) * pixel[1]);
                pixel[2] = cv::saturate_cast<uchar>(alpha * color[2] + (1 - alpha) * pixel[2]);
            }
        }
    }
    
    return overlay;
}
