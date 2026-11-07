#include <algorithm>
#include <gtest/gtest.h>

#include "generator.hpp"
#include "metrics.hpp"
#include "pipeline.hpp"
#include "test_common.hpp"

using namespace fraudnet;

TEST(Training, TinyRunLearnsSomethingAndTunesThresholdOnValidation) {
  const auto r = fixtures::trainTiny();
  EXPECT_GT(r.testMetrics.rocAuc, 0.85);
  EXPECT_DOUBLE_EQ(r.model.threshold(), r.valMetrics.threshold);
  EXPECT_DOUBLE_EQ(r.testMetrics.threshold, r.model.threshold());
  EXPECT_GT(r.model.threshold(), 0.0);
  EXPECT_LT(r.model.threshold(), 1.0);
}

TEST(Training, SameSeedGivesBitIdenticalModelDifferentSeedDoesNot) {
  const auto a = fixtures::trainTiny(7), b = fixtures::trainTiny(7), c = fixtures::trainTiny(8);
  EXPECT_EQ(a.model.network().params(), b.model.network().params());
  EXPECT_EQ(a.model.features().scaler().mean(), b.model.features().scaler().mean());
  EXPECT_DOUBLE_EQ(a.model.threshold(), b.model.threshold());
  EXPECT_NE(a.model.network().params(), c.model.network().params());
}

TEST(Training, PretrainedTestModelGeneralizesToFreshData) {
  const auto& p = fixtures::testPredictor();
  const auto& data = fixtures::evalData();
  const auto m = evaluate(scoreSamples(p, data), labelsOf(data), p.threshold());
  EXPECT_GT(m.rocAuc, 0.95);
  EXPECT_GT(m.prAuc, 0.5);
  EXPECT_GT(m.f1, 0.5);
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
  cfg.hidden = {8};
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
  const auto r = fixtures::trainTiny();
  const auto& h = r.nnTrain.history;
  ASSERT_GE(h.size(), 3u);
  EXPECT_LT(h.back().trainLoss, h.front().trainLoss);
}

TEST(Training, DivergingRunFailsLoudlyInsteadOfProducingNaNModel) {
  GeneratorConfig g;
  g.rows = 2000;
  TrainConfig cfg;
  cfg.hidden = {8};
  cfg.learningRate = 1e300;  // absurd step size -> non-finite weights
  cfg.maxEpochs = 5;
  EXPECT_THROW(trainPipeline(generateDataset(g), cfg, SplitConfig{}), std::runtime_error);
}
