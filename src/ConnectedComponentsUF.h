#pragma once
#include <numeric>

#include "IComponentDetector.h"
#include <vector>

class ConnectedComponentsUF : public IComponentDetector {
public:
    cv::Mat detect(const cv::Mat& binary, int minSize = 0) override;
    // 动态名称，指明4/8邻域
    std::string name() const override { return m_useEightConnectivity ? "Union-Find Custom (8)" : "Union-Find Custom (4)"; }
    int numComponents() const override { return m_numComponents; }

    // 选择4邻域或8邻域，默认4邻域
    void setEightConnectivity(bool enabled) { m_useEightConnectivity = enabled; }

private:
    struct UnionFind {
        std::vector<int> parent, size;
        explicit UnionFind(int n) : parent(n), size(n, 1) { iota(parent.begin(), parent.end(), 0); }
        int find(int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); }
        void unite(int a, int b) {
            a = find(a); b = find(b);
            if (a != b) {
                if (size[a] < size[b]) std::swap(a, b);
                parent[b] = a;
                size[a] += size[b];
            }
        }
    };

    int m_numComponents = 0;
    bool m_useEightConnectivity = false;
};
