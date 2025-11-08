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
        cerr << "❌ 输入必须为单通道二值图像" << endl;
        return Mat();
    }

    int rows = binary.rows, cols = binary.cols;
    Mat labels = Mat::zeros(rows, cols, CV_32S);

    int label = 0;
    // 邻域偏移：根据 4 邻域或 8 邻域选择
    static const int dx4[4] = {-1, 1, 0, 0};
    static const int dy4[4] = {0, 0, -1, 1};
    static const int dx8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static const int dy8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int* dx = useEightConnectivity ? dx8 : dx4;
    const int* dy = useEightConnectivity ? dy8 : dy4;
    const int K = useEightConnectivity ? 8 : 4;
    vector<int> compSizes;

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            if (binary.at<uchar>(i, j) == 255 && labels.at<int>(i, j) == 0) {
                label++;
                int count = 0;
                queue<Point> q;
                q.push(Point(j, i));
                labels.at<int>(i, j) = label;

                while (!q.empty()) {
                    Point p = q.front(); q.pop();
                    count++;
                    for (int k = 0; k < K; ++k) {
                        int nx = p.x + dx[k];
                        int ny = p.y + dy[k];
                        if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
                            if (binary.at<uchar>(ny, nx) == 255 && labels.at<int>(ny, nx) == 0) {
                                labels.at<int>(ny, nx) = label;
                                q.push(Point(nx, ny));
                            }
                        }
                    }
                }
                compSizes.push_back(count);
            }

    // 去除小连通域
    int newLabel = 0;
    Mat filtered = Mat::zeros(rows, cols, CV_32S);
    vector<int> valid(label + 1, 0);

    for (int i = 1; i <= label; ++i)
        if (compSizes[i - 1] >= minSize)
            valid[i] = ++newLabel;

    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            if (valid[labels.at<int>(y, x)] > 0)
                filtered.at<int>(y, x) = valid[labels.at<int>(y, x)];

    m_numComponents = newLabel;
    return filtered;
}
