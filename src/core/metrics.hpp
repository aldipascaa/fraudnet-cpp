#pragma once
#include <vector>

#include <nlohmann/json.hpp>

namespace fraudnet {

struct Confusion {
  std::size_t tp = 0, fp = 0, tn = 0, fn = 0;
};

struct Metrics {
  double threshold = 0.5;
  double precision = 0, recall = 0, f1 = 0, accuracy = 0;
  double rocAuc = 0.5, prAuc = 0;
  Confusion cm;
  std::size_t n = 0, positives = 0;
};

struct Curve {
  std::vector<double> x, y;
};

Confusion confusion(const std::vector<double>& scores, const std::vector<int>& labels, double threshold);
/// Area under the ROC curve (rank statistic, ties handled by average ranks).
double rocAuc(const std::vector<double>& scores, const std::vector<int>& labels);
/// Average precision (area under the precision-recall curve, step-wise interpolation).
double prAuc(const std::vector<double>& scores, const std::vector<int>& labels);
/// Threshold maximizing F1 (midpoint between adjacent distinct scores).
double bestF1Threshold(const std::vector<double>& scores, const std::vector<int>& labels);
Metrics evaluate(const std::vector<double>& scores, const std::vector<int>& labels, double threshold);
/// (fpr, tpr) points, at most ~maxPoints.
Curve rocCurve(const std::vector<double>& scores, const std::vector<int>& labels, std::size_t maxPoints = 200);
/// (recall, precision) points, at most ~maxPoints.
Curve prCurve(const std::vector<double>& scores, const std::vector<int>& labels, std::size_t maxPoints = 200);

nlohmann::json metricsToJson(const Metrics& m);
Metrics metricsFromJson(const nlohmann::json& j);

}  // namespace fraudnet
