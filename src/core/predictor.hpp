#pragma once
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "model.hpp"

namespace fraudnet {

struct Contribution {
  std::string feature;
  nlohmann::json value;      // raw input value of this feature
  double contribution = 0;   // change in log-odds attributable to this feature (+ = towards fraud)
};

struct Prediction {
  double probability = 0;
  bool isFraud = false;
  std::vector<Contribution> contributions;  // sorted by |contribution| descending
};

/// Immutable inference session. Holds the model behind `shared_ptr<const FraudModel>` and exposes
/// only const operations that keep all scratch state on the caller's stack, so one Predictor can
/// be copied freely and used from any number of threads concurrently without locking.
class Predictor {
 public:
  explicit Predictor(FraudModel model);
  explicit Predictor(std::shared_ptr<const FraudModel> model);

  const FraudModel& model() const { return *model_; }
  double threshold() const { return model_->threshold(); }

  /// Fraud probability. Throws std::invalid_argument for invalid input and std::runtime_error if
  /// the network produces a non-finite value.
  double probability(const Transaction& tx) const;
  /// `topK` = number of explanation entries (0 = none, large = all). Occlusion attribution:
  /// replaces one feature group by its training mean and measures the change in log-odds.
  Prediction predict(const Transaction& tx, std::size_t topK = 5) const;
  std::vector<Prediction> predictBatch(const std::vector<Transaction>& txs, std::size_t topK = 0) const;

 private:
  std::shared_ptr<const FraudModel> model_;
};

/// Scores every sample.
std::vector<double> scoreSamples(const Predictor& predictor, const std::vector<Sample>& samples);

}  // namespace fraudnet
