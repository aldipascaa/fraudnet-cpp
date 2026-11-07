#pragma once
#include <vector>

#include "features.hpp"
#include "scaler.hpp"

namespace fraudnet {

/// The one and only path from a raw Transaction to the numeric row the network consumes:
/// validate -> engineer features (features.hpp) -> standardize (fitted scaler).
///
/// Training builds its matrices with transform(samples); serving scores single transactions with
/// transform(tx). Both call the same function row by row, so training/serving skew is impossible
/// by construction (and covered by a parity test).
class FeaturePipeline {
 public:
  FeaturePipeline() = default;
  explicit FeaturePipeline(StandardScaler scaler) : scaler_(std::move(scaler)) {}

  /// Fits the scaler on the (training) samples only.
  static FeaturePipeline fit(const std::vector<Sample>& train);

  bool fitted() const { return scaler_.fitted(); }
  const StandardScaler& scaler() const { return scaler_; }

  /// Standardized feature row for one transaction. Throws std::invalid_argument for invalid input.
  std::vector<double> transform(const Transaction& tx) const;
  /// Standardized feature matrix, one row per sample.
  Matrix transform(const std::vector<Sample>& samples) const;

  /// Unscaled engineered features (what the scaler is fitted on).
  static Matrix rawFeatures(const std::vector<Sample>& samples);

 private:
  StandardScaler scaler_;
};

}  // namespace fraudnet
