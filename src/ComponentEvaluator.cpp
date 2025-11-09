#include "ComponentEvaluator.h"
#include "VisualizationUtils.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>

using namespace cv;
using namespace std;
// 基于整数标签矩阵的像素一致率计算（避免彩色图多通道导致 countNonZero 断言失败）
static Mat remapToMatch(const Mat& refLabels, const Mat& testLabels) {
    CV_Assert(refLabels.size() == testLabels.size());
    CV_Assert(refLabels.type() == CV_32S && testLabels.type() == CV_32S);
    // 统计每个 test 标签与 ref 标签的重叠像素数：counts[test][ref] -> count
    std::unordered_map<int, std::unordered_map<int, int>> counts;
    for (int y = 0; y < refLabels.rows; ++y) {
        const int* rr = refLabels.ptr<int>(y);
        const int* tt = testLabels.ptr<int>(y);
        for (int x = 0; x < refLabels.cols; ++x) {
            int r = rr[x];
            int t = tt[x];
            if (t == 0) continue; // 忽略测试图背景对映射的影响
            counts[t][r]++;
        }
    }

    // 构建 test->ref 的最佳映射（选择重叠最多的 ref 标签）
    std::unordered_map<int, int> t2r;
    t2r[0] = 0; // 背景保持 0
    for (auto& kv : counts) {
        int t = kv.first;
        int bestRef = 0;
        int bestCnt = -1;
        for (auto& kv2 : kv.second) {
            if (kv2.second > bestCnt) {
                bestCnt = kv2.second;
                bestRef = kv2.first;
            }
        }
        t2r[t] = bestRef;
    }

    Mat mapped = Mat::zeros(testLabels.size(), CV_32S);
    for (int y = 0; y < testLabels.rows; ++y) {
        const int* tt = testLabels.ptr<int>(y);
        int* mm = mapped.ptr<int>(y);
        for (int x = 0; x < testLabels.cols; ++x) {
            int t = tt[x];
            auto it = t2r.find(t);
            mm[x] = (it != t2r.end()) ? it->second : 0;
        }
    }
    return mapped;
}

static double pixelAccuracy(const Mat& refLabels, const Mat& testLabels) {
    if (refLabels.size()!=testLabels.size()) return 0.0;
    if (refLabels.type()!=testLabels.type()) return 0.0;
    // 期望 CV_32S
    CV_Assert(refLabels.type() == CV_32S);
    CV_Assert(testLabels.type() == CV_32S);
    // 先将 test 标签重映射到与 ref 尽量一致
    Mat mappedTest = remapToMatch(refLabels, testLabels);
    // 使用表达式生成单通道 8U 掩码
    Mat eqMask = (refLabels == mappedTest); // 结果为 8U 单通道
    return countNonZero(eqMask) / (double)refLabels.total();
}

// 计算 mean IoU：对每个参考组件 r，找到与其 IoU 最大的测试组件 t，取这些最大 IoU 的平均。
// 注：背景(0)不参与。
static double meanIoU(const Mat& refLabels, const Mat& testLabels) {
    CV_Assert(refLabels.size() == testLabels.size());
    CV_Assert(refLabels.type() == CV_32S && testLabels.type() == CV_32S);

    // 统计每个参考标签的像素数、每个测试标签的像素数、以及两者交集像素数
    unordered_map<int, int> refArea, testArea;
    unordered_map<long long, int> inter; // key = ((long long)r<<32) | t

    const int rows = refLabels.rows, cols = refLabels.cols;
    for (int y = 0; y < rows; ++y) {
        const int* rr = refLabels.ptr<int>(y);
        const int* tt = testLabels.ptr<int>(y);
        for (int x = 0; x < cols; ++x) {
            int r = rr[x];
            int t = tt[x];
            if (r > 0) refArea[r]++;
            if (t > 0) testArea[t]++;
            if (r > 0 && t > 0) {
                long long key = ( (long long)r << 32 ) | (unsigned int)t;
                inter[key]++;
            }
        }
    }

    if (refArea.empty()) return 0.0;

    // 对每个参考标签，计算与所有测试标签的 IoU，并取最大值
    double sumMaxIoU = 0.0;
    int count = 0;
    for (auto& kv : refArea) {
        int r = kv.first;
        int areaR = kv.second;
        double best = 0.0;
        // 遍历与 r 有交集的测试标签（通过 inter 过滤）
        // 为了高效，扫描 testArea 并通过 inter 查交
        for (auto& tv : testArea) {
            int t = tv.first;
            int areaT = tv.second;
            long long key = ( (long long)r << 32 ) | (unsigned int)t;
            auto it = inter.find(key);
            int interRT = (it != inter.end()) ? it->second : 0;
            if (interRT == 0) continue; // 无交集 IoU 为 0
            int uni = areaR + areaT - interRT;
            if (uni > 0) {
                double iou = (double)interRT / (double)uni;
                if (iou > best) best = iou;
            }
        }
        sumMaxIoU += best;
        count++;
    }
    return (count > 0) ? (sumMaxIoU / count) : 0.0;
}

std::vector<ComponentEvaluator::Result> ComponentEvaluator::evaluate(const Mat& binary, const vector<IComponentDetector*>& detectors) {
    vector<Result> results;

    for (auto det : detectors) {
        int64 t0 = getTickCount();
        Mat labels = det->detect(binary);
        double time = (getTickCount() - t0)*1000/getTickFrequency();
        results.push_back({
            det->name(),
            det->numComponents(),
            time,
            0.0,
            0.0,
            labels,
            VisualizationUtils::labelsToColorImage(labels, det->numComponents(), /*colorScheme=*/0)
        });
    }

    // 基于第一个算法的原始标签计算一致率
    for (size_t i = 1; i < results.size(); ++i) {
        results[i].pixelAccuracy = pixelAccuracy(results[0].labels, results[i].labels);
        results[i].meanIoU = meanIoU(results[0].labels, results[i].labels);
    }
    if (!results.empty()) {
        results[0].pixelAccuracy = 1.0; // 基准与自身一致率视为 100%
        results[0].meanIoU = 1.0;       // 基准与自身 mIoU 也为 1
    }
    return results;
}
