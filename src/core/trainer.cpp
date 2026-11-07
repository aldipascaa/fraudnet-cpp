#include "trainer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "metrics.hpp"
#include "util.hpp"

namespace fraudnet {

TrainResult trainMlp(const Matrix& xTrain, const std::vector<int>& yTrain, const Matrix& xVal,
                     const std::vector<int>& yVal, const TrainConfig& cfg,
                     const std::function<void(const EpochLog&)>& onEpoch) {
  if (xTrain.rows == 0 || xTrain.rows != yTrain.size()) throw std::invalid_argument("trainMlp: bad training data");
  if (xVal.rows != yVal.size()) throw std::invalid_argument("trainMlp: bad validation data");
  const auto t0 = std::chrono::steady_clock::now();

  std::vector<std::size_t> sizes{xTrain.cols};
  sizes.insert(sizes.end(), cfg.hidden.begin(), cfg.hidden.end());
  sizes.push_back(1);

  TrainResult res;
  res.net = Mlp(sizes, cfg.seed);
  std::size_t pos = 0;
  for (int y : yTrain) pos += (y == 1);
  const std::size_t neg = yTrain.size() - pos;
  res.posWeight = cfg.posWeight >= 0 ? cfg.posWeight
                                     : (pos > 0 ? std::sqrt(static_cast<double>(neg) / static_cast<double>(pos)) : 1.0);

  Mlp& net = res.net;
  const std::size_t P = net.numParams();
  std::vector<double> m(P, 0.0), v(P, 0.0), grad;
  constexpr double b1 = 0.9, b2 = 0.999, eps = 1e-8;
  std::size_t step = 0;

  std::vector<std::size_t> order(xTrain.rows);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::vector<std::size_t> valIdx(xVal.rows);
  std::iota(valIdx.begin(), valIdx.end(), std::size_t{0});
  Rng rng(cfg.seed ^ 0xA5A5A5A5ULL);

  double bestScore = -1.0;
  std::vector<double> bestParams = net.params();
  std::size_t sinceBest = 0;

  for (std::size_t epoch = 1; epoch <= cfg.maxEpochs; ++epoch) {
    rng.shuffle(order);
    double lossSum = 0;
    std::size_t batches = 0;
    for (std::size_t start = 0; start < order.size(); start += cfg.batchSize) {
      const std::size_t end = std::min(order.size(), start + cfg.batchSize);
      const std::vector<std::size_t> batch(order.begin() + static_cast<std::ptrdiff_t>(start),
                                           order.begin() + static_cast<std::ptrdiff_t>(end));
      lossSum += net.lossAndGrad(xTrain, yTrain, batch, res.posWeight, cfg.l2, grad);
      ++step;
      const double c1 = 1.0 - std::pow(b1, static_cast<double>(step));
      const double c2 = 1.0 - std::pow(b2, static_cast<double>(step));
      auto& w = net.params();
      for (std::size_t k = 0; k < P; ++k) {
        m[k] = b1 * m[k] + (1 - b1) * grad[k];
        v[k] = b2 * v[k] + (1 - b2) * grad[k] * grad[k];
        w[k] -= cfg.learningRate * (m[k] / c1) / (std::sqrt(v[k] / c2) + eps);
      }
      ++batches;
    }

    EpochLog log;
    log.epoch = epoch;
    log.trainLoss = lossSum / static_cast<double>(std::max<std::size_t>(1, batches));
    if (!std::isfinite(log.trainLoss))
      throw std::runtime_error("training diverged (non-finite loss at epoch " + std::to_string(epoch) + "); try a lower --lr");
    double score = static_cast<double>(epoch) * 1e-12;  // no validation data: keep the latest epoch
    if (xVal.rows > 0) {
      const auto p = net.predictProba(xVal);
      log.valLoss = net.loss(xVal, yVal, valIdx, res.posWeight, 0.0);
      log.valPrAuc = prAuc(p, yVal);
      log.valRocAuc = rocAuc(p, yVal);
      score = log.valPrAuc;
    }
    res.history.push_back(log);
    if (onEpoch) onEpoch(log);

    if (score > bestScore) {
      bestScore = score;
      bestParams = net.params();
      res.bestEpoch = epoch;
      sinceBest = 0;
    } else if (++sinceBest >= cfg.patience) {
      break;
    }
  }
  net.params() = bestParams;
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return res;
}

}  // namespace fraudnet
