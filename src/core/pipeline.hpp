#pragma once
#include <functional>
#include <string>
#include <vector>

#include "metrics.hpp"
#include "model.hpp"
#include "trainer.hpp"

namespace fraudnet {

struct SplitConfig {
  double valFraction = 0.15;
  double testFraction = 0.15;
  std::uint64_t seed = 42;
};

struct DataSplit {
  std::vector<Sample> train, val, test;
};

/// Stratified shuffle split (keeps the fraud rate in every part).
DataSplit stratifiedSplit(const std::vector<Sample>& all, const SplitConfig& cfg);

std::vector<int> labelsOf(const std::vector<Sample>& samples);

struct PipelineResult {
  FraudModel model;                  // neural network + scaler + tuned threshold + metadata
  Metrics valMetrics, testMetrics;   // neural network
  Metrics baselineVal, baselineTest; // logistic regression baseline
  TrainResult nnTrain, lrTrain;
  Curve roc, pr;                     // network, test set
  DataSplit split;
};

/// Full training pipeline: split -> fit scaler on train -> train logistic regression baseline and
/// the neural network (early stopping) -> tune F1 thresholds on validation -> evaluate on test.
PipelineResult trainPipeline(const std::vector<Sample>& data, const TrainConfig& cfg, const SplitConfig& split,
                             const std::function<void(const std::string& model, const EpochLog&)>& onEpoch = {});

}  // namespace fraudnet
