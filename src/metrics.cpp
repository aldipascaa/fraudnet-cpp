#include "metrics.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace fraudnet {

namespace {
void check(const std::vector<double>& s, const std::vector<int>& y) {
  if (s.size() != y.size()) throw std::invalid_argument("metrics: scores/labels size mismatch");
}
double safeDiv(double a, double b) { return b > 0 ? a / b : 0.0; }

// Indices sorted by descending score.
std::vector<std::size_t> orderDesc(const std::vector<double>& s) {
  std::vector<std::size_t> idx(s.size());
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return s[a] > s[b]; });
  return idx;
}

Curve downsample(const Curve& c, std::size_t maxPoints) {
  if (c.x.size() <= maxPoints || maxPoints < 3) return c;
  Curve out;
  const double step = static_cast<double>(c.x.size() - 1) / static_cast<double>(maxPoints - 1);
  for (std::size_t k = 0; k < maxPoints; ++k) {
    const auto i = static_cast<std::size_t>(static_cast<double>(k) * step + 0.5);
    out.x.push_back(c.x[std::min(i, c.x.size() - 1)]);
    out.y.push_back(c.y[std::min(i, c.x.size() - 1)]);
  }
  return out;
}
}  // namespace

Confusion confusion(const std::vector<double>& scores, const std::vector<int>& labels, double threshold) {
  check(scores, labels);
  Confusion c;
  for (std::size_t i = 0; i < scores.size(); ++i) {
    const bool pred = scores[i] >= threshold;
    const bool act = labels[i] == 1;
    if (pred && act) ++c.tp;
    else if (pred) ++c.fp;
    else if (act) ++c.fn;
    else ++c.tn;
  }
  return c;
}

double rocAuc(const std::vector<double>& scores, const std::vector<int>& labels) {
  check(scores, labels);
  const std::size_t n = scores.size();
  std::vector<std::size_t> idx(n);
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return scores[a] < scores[b]; });
  double rankSumPos = 0;
  std::size_t nPos = 0;
  for (std::size_t i = 0; i < n;) {
    std::size_t j = i;
    while (j < n && scores[idx[j]] == scores[idx[i]]) ++j;
    const double avgRank = 0.5 * (static_cast<double>(i + 1) + static_cast<double>(j));  // 1-based
    for (std::size_t k = i; k < j; ++k)
      if (labels[idx[k]] == 1) {
        rankSumPos += avgRank;
        ++nPos;
      }
    i = j;
  }
  const std::size_t nNeg = n - nPos;
  if (nPos == 0 || nNeg == 0) return 0.5;
  const double p = static_cast<double>(nPos), q = static_cast<double>(nNeg);
  return (rankSumPos - p * (p + 1.0) / 2.0) / (p * q);
}

double prAuc(const std::vector<double>& scores, const std::vector<int>& labels) {
  check(scores, labels);
  const auto idx = orderDesc(scores);
  std::size_t P = 0;
  for (int l : labels) P += (l == 1);
  if (P == 0) return 0.0;
  double tp = 0, fp = 0, prevRecall = 0, ap = 0;
  for (std::size_t i = 0; i < idx.size();) {
    std::size_t j = i;
    while (j < idx.size() && scores[idx[j]] == scores[idx[i]]) {
      (labels[idx[j]] == 1 ? tp : fp) += 1;
      ++j;
    }
    const double recall = tp / static_cast<double>(P);
    const double precision = tp / (tp + fp);
    ap += (recall - prevRecall) * precision;
    prevRecall = recall;
    i = j;
  }
  return ap;
}

double bestF1Threshold(const std::vector<double>& scores, const std::vector<int>& labels) {
  check(scores, labels);
  const auto idx = orderDesc(scores);
  std::size_t P = 0;
  for (int l : labels) P += (l == 1);
  if (P == 0 || scores.empty()) return 0.5;
  double tp = 0, fp = 0, bestF1 = -1, bestThr = 0.5;
  for (std::size_t i = 0; i < idx.size();) {
    std::size_t j = i;
    while (j < idx.size() && scores[idx[j]] == scores[idx[i]]) {
      (labels[idx[j]] == 1 ? tp : fp) += 1;
      ++j;
    }
    const double precision = tp / (tp + fp);
    const double recall = tp / static_cast<double>(P);
    const double f1 = safeDiv(2 * precision * recall, precision + recall);
    if (f1 > bestF1) {
      bestF1 = f1;
      const double cur = scores[idx[i]];
      bestThr = (j < idx.size()) ? 0.5 * (cur + scores[idx[j]]) : cur;
    }
    i = j;
  }
  return bestThr;
}

Metrics evaluate(const std::vector<double>& scores, const std::vector<int>& labels, double threshold) {
  Metrics m;
  m.threshold = threshold;
  m.cm = confusion(scores, labels, threshold);
  const double tp = static_cast<double>(m.cm.tp), fp = static_cast<double>(m.cm.fp);
  const double fn = static_cast<double>(m.cm.fn), tn = static_cast<double>(m.cm.tn);
  m.precision = safeDiv(tp, tp + fp);
  m.recall = safeDiv(tp, tp + fn);
  m.f1 = safeDiv(2 * m.precision * m.recall, m.precision + m.recall);
  m.accuracy = safeDiv(tp + tn, tp + tn + fp + fn);
  m.rocAuc = rocAuc(scores, labels);
  m.prAuc = prAuc(scores, labels);
  m.n = scores.size();
  m.positives = m.cm.tp + m.cm.fn;
  return m;
}

Curve rocCurve(const std::vector<double>& scores, const std::vector<int>& labels, std::size_t maxPoints) {
  check(scores, labels);
  const auto idx = orderDesc(scores);
  std::size_t P = 0;
  for (int l : labels) P += (l == 1);
  const std::size_t N = labels.size() - P;
  Curve c;
  c.x.push_back(0);
  c.y.push_back(0);
  double tp = 0, fp = 0;
  for (std::size_t i = 0; i < idx.size();) {
    std::size_t j = i;
    while (j < idx.size() && scores[idx[j]] == scores[idx[i]]) {
      (labels[idx[j]] == 1 ? tp : fp) += 1;
      ++j;
    }
    c.x.push_back(safeDiv(fp, static_cast<double>(N)));
    c.y.push_back(safeDiv(tp, static_cast<double>(P)));
    i = j;
  }
  return downsample(c, maxPoints);
}

Curve prCurve(const std::vector<double>& scores, const std::vector<int>& labels, std::size_t maxPoints) {
  check(scores, labels);
  const auto idx = orderDesc(scores);
  std::size_t P = 0;
  for (int l : labels) P += (l == 1);
  Curve c;
  double tp = 0, fp = 0;
  for (std::size_t i = 0; i < idx.size();) {
    std::size_t j = i;
    while (j < idx.size() && scores[idx[j]] == scores[idx[i]]) {
      (labels[idx[j]] == 1 ? tp : fp) += 1;
      ++j;
    }
    c.x.push_back(safeDiv(tp, static_cast<double>(P)));
    c.y.push_back(tp / (tp + fp));
    i = j;
  }
  return downsample(c, maxPoints);
}

nlohmann::json metricsToJson(const Metrics& m) {
  return {{"threshold", m.threshold}, {"precision", m.precision}, {"recall", m.recall},
          {"f1", m.f1},               {"accuracy", m.accuracy},   {"roc_auc", m.rocAuc},
          {"pr_auc", m.prAuc},        {"samples", m.n},           {"positives", m.positives},
          {"confusion_matrix", {{"tp", m.cm.tp}, {"fp", m.cm.fp}, {"tn", m.cm.tn}, {"fn", m.cm.fn}}}};
}

Metrics metricsFromJson(const nlohmann::json& j) {
  Metrics m;
  m.threshold = j.at("threshold");
  m.precision = j.at("precision");
  m.recall = j.at("recall");
  m.f1 = j.at("f1");
  m.accuracy = j.at("accuracy");
  m.rocAuc = j.at("roc_auc");
  m.prAuc = j.at("pr_auc");
  m.n = j.at("samples");
  m.positives = j.at("positives");
  const auto& c = j.at("confusion_matrix");
  m.cm.tp = c.at("tp");
  m.cm.fp = c.at("fp");
  m.cm.tn = c.at("tn");
  m.cm.fn = c.at("fn");
  return m;
}

}  // namespace fraudnet
