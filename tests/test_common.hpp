#pragma once
#include "generator.hpp"
#include "pipeline.hpp"
#include "predictor.hpp"

namespace fraudnet::fixtures {

/// Path of the tiny pre-trained model checked into the repo. Tests run with the source directory
/// as working directory (see CMakeLists.txt), so no path is baked into the test binary.
inline constexpr const char* kTestModelPath = "tests/data/test_model.json";

/// Pre-trained model loaded once per process: fast and independent of the training code, so
/// inference/API tests neither slow down nor break when training changes.
inline const Predictor& testPredictor() {
  static const Predictor p(FraudModel::load(kTestModelPath));
  return p;
}

/// Fresh labelled data (a different seed than anything the test model was trained on).
inline const std::vector<Sample>& evalData() {
  static const std::vector<Sample> d = [] {
    GeneratorConfig g;
    g.rows = 20000;
    g.seed = 777;
    return generateDataset(g);
  }();
  return d;
}

/// A deliberately tiny, fast training run (a fraction of a second) for tests of the training code.
inline PipelineResult trainTiny(std::uint64_t seed = 7) {
  GeneratorConfig g;
  g.rows = 5000;
  g.seed = 123;
  TrainConfig t;
  t.hidden = {8};
  t.maxEpochs = 6;
  t.patience = 3;
  t.seed = seed;
  SplitConfig s;
  s.seed = seed;
  return trainPipeline(generateDataset(g), t, s);
}

inline Transaction suspiciousTx() {
  Transaction t;
  t.amount = 900;
  t.hour = 3;
  t.merchant_category = "electronics";
  t.distance_from_home_km = 800;
  t.distance_from_last_txn_km = 400;
  t.txn_count_1h = 4;
  t.txn_count_24h = 8;
  t.avg_amount_30d = 45;
  t.card_present = false;
  t.is_foreign = true;
  t.account_age_days = 20;
  return t;
}

inline Transaction typicalTx() {
  Transaction t;
  t.amount = 40;
  t.hour = 13;
  t.merchant_category = "grocery";
  t.distance_from_home_km = 3;
  t.distance_from_last_txn_km = 2;
  t.avg_amount_30d = 55;
  t.account_age_days = 900;
  return t;
}

}  // namespace fraudnet::fixtures
