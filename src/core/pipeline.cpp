#include "pipeline.hpp"

#include <cmath>
#include <stdexcept>

#include "util.hpp"
#include "fraudnet/version.hpp"

namespace fraudnet {

DataSplit stratifiedSplit(const std::vector<Sample>& all, const SplitConfig& cfg) {
  if (cfg.valFraction < 0 || cfg.testFraction < 0 || cfg.valFraction + cfg.testFraction >= 1.0)
    throw std::invalid_argument("invalid split fractions");
  std::vector<std::size_t> byClass[2];
  for (std::size_t i = 0; i < all.size(); ++i) byClass[all[i].label == 1 ? 1 : 0].push_back(i);
  Rng rng(cfg.seed);
  DataSplit out;
  for (auto& idx : byClass) {
    rng.shuffle(idx);
    const auto n = static_cast<double>(idx.size());
    const auto nTest = static_cast<std::size_t>(std::llround(n * cfg.testFraction));
    const auto nVal = static_cast<std::size_t>(std::llround(n * cfg.valFraction));
    for (std::size_t k = 0; k < idx.size(); ++k) {
      auto& dst = k < nTest ? out.test : (k < nTest + nVal ? out.val : out.train);
      dst.push_back(all[idx[k]]);
    }
  }
  // Interleave classes again so batches are not ordered by class.
  Rng mix(cfg.seed + 1);
  mix.shuffle(out.train);
  mix.shuffle(out.val);
  mix.shuffle(out.test);
  return out;
}

std::vector<int> labelsOf(const std::vector<Sample>& samples) {
  std::vector<int> y;
  y.reserve(samples.size());
  for (const auto& s : samples) y.push_back(s.label);
  return y;
}

PipelineResult trainPipeline(const std::vector<Sample>& data, const TrainConfig& cfg, const SplitConfig& splitCfg,
                             const std::function<void(const std::string&, const EpochLog&)>& onEpoch) {
  PipelineResult r;
  r.split = stratifiedSplit(data, splitCfg);
  const auto& sp = r.split;
  if (sp.train.empty() || sp.val.empty() || sp.test.empty()) throw std::runtime_error("dataset too small to split");

  // The same FeaturePipeline instance is stored in the model, so serving cannot drift from training.
  const FeaturePipeline features = FeaturePipeline::fit(sp.train);
  const Matrix xTrain = features.transform(sp.train);
  const Matrix xVal = features.transform(sp.val);
  const Matrix xTest = features.transform(sp.test);
  const auto yTrain = labelsOf(sp.train), yVal = labelsOf(sp.val), yTest = labelsOf(sp.test);

  // Baseline: logistic regression (no hidden layers).
  TrainConfig lrCfg = cfg;
  lrCfg.hidden.clear();
  r.lrTrain = trainMlp(xTrain, yTrain, xVal, yVal, lrCfg, [&](const EpochLog& e) {
    if (onEpoch) onEpoch("logreg", e);
  });
  {
    const auto pv = r.lrTrain.net.predictProba(xVal);
    const double thr = bestF1Threshold(pv, yVal);
    r.baselineVal = evaluate(pv, yVal, thr);
    r.baselineTest = evaluate(r.lrTrain.net.predictProba(xTest), yTest, thr);
  }

  r.nnTrain = trainMlp(xTrain, yTrain, xVal, yVal, cfg, [&](const EpochLog& e) {
    if (onEpoch) onEpoch("mlp", e);
  });
  const auto pv = r.nnTrain.net.predictProba(xVal);
  const double thr = bestF1Threshold(pv, yVal);  // tuned on validation only
  r.valMetrics = evaluate(pv, yVal, thr);
  const auto pt = r.nnTrain.net.predictProba(xTest);
  r.testMetrics = evaluate(pt, yTest, thr);
  r.roc = rocCurve(pt, yTest, 120);
  r.pr = prCurve(pt, yTest, 120);

  r.model = FraudModel(features, r.nnTrain.net, thr);
  std::size_t fraud = 0;
  for (const auto& s : data) fraud += (s.label == 1);

  nlohmann::json hist = nlohmann::json::array();
  for (const auto& e : r.nnTrain.history)
    hist.push_back({{"epoch", e.epoch}, {"train_loss", e.trainLoss}, {"val_loss", e.valLoss}, {"val_pr_auc", e.valPrAuc}});

  auto& meta = r.model.meta();
  meta["name"] = "FraudNet";
  meta["version"] = FRAUDNET_VERSION;
  meta["created_at"] = isoTimestampUtc();
  meta["training"] = {
      {"rows", data.size()},
      {"fraud_rows", fraud},
      {"fraud_rate", data.empty() ? 0.0 : static_cast<double>(fraud) / static_cast<double>(data.size())},
      {"train_rows", sp.train.size()},
      {"val_rows", sp.val.size()},
      {"test_rows", sp.test.size()},
      {"split", {{"val_fraction", splitCfg.valFraction}, {"test_fraction", splitCfg.testFraction}, {"seed", splitCfg.seed}}},
      {"hidden_layers", cfg.hidden},
      {"learning_rate", cfg.learningRate},
      {"batch_size", cfg.batchSize},
      {"l2", cfg.l2},
      {"seed", cfg.seed},
      {"positive_class_weight", r.nnTrain.posWeight},
      {"epochs_run", r.nnTrain.history.size()},
      {"best_epoch", r.nnTrain.bestEpoch},
      {"seconds", r.nnTrain.seconds},
      {"optimizer", "adam"},
      {"history", hist}};
  meta["metrics"] = {{"validation", metricsToJson(r.valMetrics)},
                     {"test", metricsToJson(r.testMetrics)},
                     {"baseline_logreg_validation", metricsToJson(r.baselineVal)},
                     {"baseline_logreg_test", metricsToJson(r.baselineTest)},
                     {"roc_curve", {{"fpr", r.roc.x}, {"tpr", r.roc.y}}},
                     {"pr_curve", {{"recall", r.pr.x}, {"precision", r.pr.y}}}};
  return r;
}

}  // namespace fraudnet
