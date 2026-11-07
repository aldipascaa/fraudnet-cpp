#include <gtest/gtest.h>

#include "metrics.hpp"

using namespace fraudnet;

TEST(Metrics, ConfusionAndDerivedScoresOnKnownInput) {
  const std::vector<double> s{0.1, 0.4, 0.35, 0.8};
  const std::vector<int> y{0, 0, 1, 1};
  const auto m = evaluate(s, y, 0.5);
  EXPECT_EQ(m.cm.tp, 1u);
  EXPECT_EQ(m.cm.fp, 0u);
  EXPECT_EQ(m.cm.tn, 2u);
  EXPECT_EQ(m.cm.fn, 1u);
  EXPECT_DOUBLE_EQ(m.precision, 1.0);
  EXPECT_DOUBLE_EQ(m.recall, 0.5);
  EXPECT_NEAR(m.f1, 2.0 / 3.0, 1e-12);
  EXPECT_DOUBLE_EQ(m.accuracy, 0.75);
}

TEST(Metrics, RocAucKnownValues) {
  EXPECT_NEAR(rocAuc({0.1, 0.4, 0.35, 0.8}, {0, 0, 1, 1}), 0.75, 1e-12);
  EXPECT_DOUBLE_EQ(rocAuc({0.1, 0.2, 0.8, 0.9}, {0, 0, 1, 1}), 1.0);
  EXPECT_DOUBLE_EQ(rocAuc({0.9, 0.8, 0.2, 0.1}, {0, 0, 1, 1}), 0.0);
  EXPECT_DOUBLE_EQ(rocAuc({0.5, 0.5, 0.5, 0.5}, {0, 1, 0, 1}), 0.5);  // ties
  EXPECT_DOUBLE_EQ(rocAuc({0.3, 0.6}, {0, 0}), 0.5);                  // degenerate
}

TEST(Metrics, PrAucKnownValues) {
  // ranking: 0.8(+) 0.4(-) 0.35(+) 0.1(-)  => AP = 0.5*1 + 0.5*(2/3)
  EXPECT_NEAR(prAuc({0.1, 0.4, 0.35, 0.8}, {0, 0, 1, 1}), 0.5 + 0.5 * 2.0 / 3.0, 1e-12);
  EXPECT_DOUBLE_EQ(prAuc({0.1, 0.2, 0.8, 0.9}, {0, 0, 1, 1}), 1.0);
  // all tied: precision equals the prevalence
  EXPECT_NEAR(prAuc({0.5, 0.5, 0.5, 0.5}, {1, 0, 0, 0}), 0.25, 1e-12);
}

TEST(Metrics, BestF1ThresholdMaximizesF1) {
  const std::vector<double> s{0.1, 0.4, 0.35, 0.8};
  const std::vector<int> y{0, 0, 1, 1};
  const double thr = bestF1Threshold(s, y);
  EXPECT_NEAR(thr, 0.225, 1e-12);  // flags 0.8, 0.4, 0.35 => P=2/3, R=1, F1=0.8
  const auto best = evaluate(s, y, thr);
  EXPECT_NEAR(best.f1, 0.8, 1e-12);
  for (double t : {0.05, 0.2, 0.3, 0.375, 0.6, 0.9}) EXPECT_LE(evaluate(s, y, t).f1, best.f1 + 1e-12);
}

TEST(Metrics, RocCurveEndpointsAndMonotone) {
  const std::vector<double> s{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2};
  const std::vector<int> y{1, 0, 1, 0, 1, 0, 0, 0};
  const auto c = rocCurve(s, y);
  EXPECT_DOUBLE_EQ(c.x.front(), 0.0);
  EXPECT_DOUBLE_EQ(c.y.front(), 0.0);
  EXPECT_DOUBLE_EQ(c.x.back(), 1.0);
  EXPECT_DOUBLE_EQ(c.y.back(), 1.0);
  for (std::size_t i = 1; i < c.x.size(); ++i) {
    EXPECT_GE(c.x[i], c.x[i - 1]);
    EXPECT_GE(c.y[i], c.y[i - 1]);
  }
  // trapezoid area must equal the rank-based AUC
  double area = 0;
  for (std::size_t i = 1; i < c.x.size(); ++i) area += (c.x[i] - c.x[i - 1]) * (c.y[i] + c.y[i - 1]) / 2;
  EXPECT_NEAR(area, rocAuc(s, y), 1e-12);
}

TEST(Metrics, DownsampledCurveKeepsEndpoints) {
  std::vector<double> s;
  std::vector<int> y;
  for (int i = 0; i < 1000; ++i) {
    s.push_back(i / 1000.0);
    y.push_back(i % 7 == 0 ? 1 : 0);
  }
  const auto c = rocCurve(s, y, 50);
  EXPECT_LE(c.x.size(), 50u);
  EXPECT_DOUBLE_EQ(c.x.back(), 1.0);
}

TEST(Metrics, JsonRoundTrip) {
  const auto m = evaluate({0.1, 0.4, 0.35, 0.8}, {0, 0, 1, 1}, 0.3);
  const auto r = metricsFromJson(metricsToJson(m));
  EXPECT_DOUBLE_EQ(r.f1, m.f1);
  EXPECT_EQ(r.cm.tp, m.cm.tp);
  EXPECT_DOUBLE_EQ(r.rocAuc, m.rocAuc);
}

TEST(Metrics, SizeMismatchThrows) {
  EXPECT_THROW(rocAuc({0.1}, {0, 1}), std::invalid_argument);
}
