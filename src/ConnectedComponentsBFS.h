#pragma once
#include "IComponentDetector.h"
#include <vector>

class ConnectedComponentsBFS : public IComponentDetector {
public:
    cv::Mat detect(const cv::Mat& binary, int minSize = 0, bool useEightConnectivity = false);
    cv::Mat detect(const cv::Mat& binary, int minSize = 0) override { return detect(binary, minSize, m_useEightConnectivity); }
    void setEightConnectivity(bool enabled) { m_useEightConnectivity = enabled; }
    std::string name() const override;
    int numComponents() const override { return m_numComponents; }

private:
    int m_numComponents = 0;
    bool m_useEightConnectivity = false;
};
