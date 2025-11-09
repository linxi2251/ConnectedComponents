#include "ConnectedComponentsBFS.h"
#include <queue>
#include <iostream>


using namespace cv;
using namespace std;

std::string ConnectedComponentsBFS::name() const {
    return m_useEightConnectivity ? "BFS Custom (8)" : "BFS Custom (4)";
}

Mat ConnectedComponentsBFS::detect(const cv::Mat& binary, int minSize, bool useEightConnectivity) {
    if (binary.empty() || binary.type() != CV_8UC1) {
        cerr << "输入必须为单通道二值图像" << endl;
        return Mat();
    }

    int rows = binary.rows, cols = binary.cols;
    Mat labels = Mat::zeros(rows, cols, CV_32S);

    // 邻域偏移：根据 4 邻域或 8 邻域选择
    static const int dx4[4] = {-1, 1, 0, 0};
    static const int dy4[4] = {0, 0, -1, 1};
    static const int dx8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static const int dy8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int* dx = useEightConnectivity ? dx8 : dx4;
    const int* dy = useEightConnectivity ? dy8 : dy4;
    const int K = useEightConnectivity ? 8 : 4;

    vector<int> compSizes;
    compSizes.reserve(1000); // 预分配空间，减少realloc

    vector<Point> queue(rows * cols);
    int head = 0, tail = 0;

    // 第一遍扫描：标记所有连通域
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (binary.at<uchar>(y, x) == 255 && labels.at<int>(y, x) == 0) {
                // 发现新连通域
                int currentLabel = (int)compSizes.size() + 1;
                int count = 0;

                head = tail = 0;  // 重置队列
                queue[tail++] = Point(x, y);
                labels.at<int>(y, x) = currentLabel;

                while (head < tail) {
                    Point p = queue[head++];  // 出队
                    count++;

                    // 遍历邻域
                    for (int k = 0; k < K; ++k) {
                        int nx = p.x + dx[k];
                        int ny = p.y + dy[k];

                        // 边界检查
                        if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
                            if (binary.at<uchar>(ny, nx) == 255 && labels.at<int>(ny, nx) == 0) {
                                labels.at<int>(ny, nx) = currentLabel;
                                queue[tail++] = Point(nx, ny);  // 入队
                            }
                        }
                    }
                }

                compSizes.push_back(count);
            }
        }
    }

    // 第二遍扫描：过滤小连通域并重新映射标签
    int newLabel = 0;
    Mat filtered = Mat::zeros(rows, cols, CV_32S);
    vector<int> valid(compSizes.size() + 1, 0);

    for (size_t i = 0; i < compSizes.size(); ++i) {
        if (compSizes[i] >= minSize) {
            valid[i + 1] = ++newLabel;  // 旧标签 -> 新标签
        }
    }

    // 应用过滤和标签映射
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int oldLabel = labels.at<int>(y, x);
            if (oldLabel > 0 && valid[oldLabel] > 0) {
                filtered.at<int>(y, x) = valid[oldLabel];
            }
        }
    }

    m_numComponents = newLabel;
    return filtered;
}
