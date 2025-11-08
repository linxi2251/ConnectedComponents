#include "ConnectedComponentsUF.h"
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

cv::Mat ConnectedComponentsUF::detect(const cv::Mat& binary, int minSize) {
    if (binary.empty() || binary.type() != CV_8UC1) {
        cerr << "❌ 输入必须为单通道二值图像" << endl;
        return Mat();
    }

    int rows = binary.rows, cols = binary.cols;
    int total = rows * cols;
    UnionFind uf(total);

    auto index = [cols](int y, int x) { return y * cols + x; };

    // 根据邻域类型进行合并：只向右、下（以及对角）方向，避免重复合并
    if (!m_useEightConnectivity) {
        // 4 邻域：只合并 (x+1,y) 和 (x,y+1)
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (binary.at<uchar>(y, x) != 255) continue;
                if (x + 1 < cols && binary.at<uchar>(y, x + 1) == 255)
                    uf.unite(index(y, x), index(y, x + 1));
                if (y + 1 < rows && binary.at<uchar>(y + 1, x) == 255)
                    uf.unite(index(y, x), index(y + 1, x));
            }
        }
    } else {
        // 8 邻域：向右、下三个方向 + 两个对角方向 (x+1,y), (x,y+1), (x+1,y+1), (x-1,y+1)
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (binary.at<uchar>(y, x) != 255) continue;
                // 右
                if (x + 1 < cols && binary.at<uchar>(y, x + 1) == 255)
                    uf.unite(index(y, x), index(y, x + 1));
                // 下
                if (y + 1 < rows && binary.at<uchar>(y + 1, x) == 255)
                    uf.unite(index(y, x), index(y + 1, x));
                // 右下
                if (x + 1 < cols && y + 1 < rows && binary.at<uchar>(y + 1, x + 1) == 255)
                    uf.unite(index(y, x), index(y + 1, x + 1));
                // 左下
                if (x - 1 >= 0 && y + 1 < rows && binary.at<uchar>(y + 1, x - 1) == 255)
                    uf.unite(index(y, x), index(y + 1, x - 1));
            }
        }
    }

    // 统计每个根的尺寸
    vector<int> componentSize(total, 0);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (binary.at<uchar>(y, x) == 255) {
                int root = uf.find(index(y, x));
                componentSize[root]++;
            }
        }
    }

    // 给满足 minSize 的根重新映射 label
    map<int, int> labelMap;
    Mat labels = Mat::zeros(rows, cols, CV_32S);
    int nextLabel = 0;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (binary.at<uchar>(y, x) == 255) {
                int root = uf.find(index(y, x));
                if (componentSize[root] >= minSize) {
                    if (labelMap[root] == 0) {
                        // 分配新标签
                        labelMap[root] = ++nextLabel;
                    }
                    labels.at<int>(y, x) = labelMap[root];
                }
            }
        }
    }

    m_numComponents = nextLabel;
    return labels;
}
