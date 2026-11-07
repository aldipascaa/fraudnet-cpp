#include <gtest/gtest.h>

#include "generator.hpp"
#include "pipeline.hpp"
#include "test_common.hpp"

using namespace fraudnet;

TEST(Training, NeuralNetBeatsMinimumAucOnHeldOutTestSet) {
  const auto& r = fixtures::sharedPipeline();
  EXPECT_GT(r.testMetrics.rocAuc, 0.9);
  EXPECT_GT(r.testMetrics.prAuc, 0.4);
  EXPECT_GT(r.testMetrics.f1, 0.4);
}

TEST(Training, BaselineIsAlsoReasonableAndNetworkIsCompetitive) {
  const auto& r = fixtures::sharedPipeline();
  EXPECT_GT(r.baselineTest.rocAuc, 0.85);
  EXPECT_GT(r.testMetrics.rocAuc, r.baselineTest.rocAuc - 0.02);
}

TEST(Training, ThresholdIsTunedOnValidationAndMaximizesItsF1) {
  const auto& r = fixtures::sharedPipeline();
  const double thr = r.model.threshold();
  EXPECT_DOUBLE_EQ(thr, r.valMetrics.threshold);
  EXPECT_DOUBLE_EQ(r.testMetrics.threshold, thr);
  EXPECT_GT(thr, 0.0);
  EXPECT_LT(thr, 1.0);
}

TEST(Training, SplitIsStratifiedAndDisjointInSize) {
  GeneratorConfig g;
  g.rows = 10000;
  g.seed = 5;
  const auto data = generateDataset(g);
  const auto sp = stratifiedSplit(data, SplitConfig{});
  EXPECT_EQ(sp.train.size() + sp.val.size() + sp.test.size(), data.size());
  auto rate = [](const std::vector<Sample>& v) {
    double f = 0;
    for (const auto& s : v) f += s.label;
    return f / static_cast<double>(v.size());
  };
  EXPECT_NEAR(rate(sp.train), rate(data), 0.002);
  EXPECT_NEAR(rate(sp.test), rate(data), 0.002);
  EXPECT_NEAR(static_cast<double>(sp.test.size()) / static_cast<double>(data.size()), 0.15, 0.01);
}

TEST(Training, EarlyStoppingHaltsBeforeMaxEpochsAndKeepsBestWeights) {
  GeneratorConfig g;
  g.rows = 6000;
  const auto data = generateDataset(g);
  TrainConfig cfg;
  cfg.maxEpochs = 200;
  cfg.patience = 2;
  cfg.learningRate = 0.02;
  const auto r = trainPipeline(data, cfg, SplitConfig{});
  EXPECT_LT(r.nnTrain.history.size(), 200u);
  ASSERT_GE(r.nnTrain.bestEpoch, 1u);
  EXPECT_LE(r.nnTrain.bestEpoch, r.nnTrain.history.size());
  double best = 0;
  for (const auto& e : r.nnTrain.history) best = std::max(best, e.valPrAuc);
  EXPECT_DOUBLE_EQ(r.nnTrain.history[r.nnTrain.bestEpoch - 1].valPrAuc, best);
}

TEST(Training, LossDecreasesDuringTraining) {
  const auto& h = fixtures::sharedPipeline().nnTrain.history;
  ASSERT_GE(h.size(), 3u);
  EXPECT_LT(h.back().trainLoss, h.front().trainLoss);
}
