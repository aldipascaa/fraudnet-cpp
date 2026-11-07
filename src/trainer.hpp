#pragma once
#include <cstdint>
#include <functional>
#include <vector>

#include "mlp.hpp"

namespace fraudnet {

struct TrainConfig {
  std::vector<std::size_t> hidden{32, 16};  // empty => logistic regression
  double learningRate = 0.003;
  std::size_t batchSize = 256;
  std::size_t maxEpochs = 60;
  std::size_t patience = 8;     // early stopping: epochs without validation PR-AUC improvement
  double posWeight = -1.0;      // < 0 => automatic: sqrt(#neg / #pos)
  double l2 = 1e-5;
  std::uint64_t seed = 42;
};

struct EpochLog {
  std::size_t epoch = 0;
  double trainLoss = 0, valLoss = 0, valPrAuc = 0, valRocAuc = 0;
};

struct TrainResult {
  Mlp net;  // weights of the best epoch (by validation PR-AUC)
  std::vector<EpochLog> history;
  std::size_t bestEpoch = 0;
  double posWeight = 1.0;
  double seconds = 0;
};

/// Mini-batch Adam with class-weighted BCE and early stopping on validation PR-AUC.
/// Inputs must already be standardized.
TrainResult trainMlp(const Matrix& xTrain, const std::vector<int>& yTrain, const Matrix& xVal,
                     const std::vector<int>& yVal, const TrainConfig& cfg,
                     const std::function<void(const EpochLog&)>& onEpoch = {});

}  // namespace fraudnet
