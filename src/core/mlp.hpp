#pragma once
#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

#include "scaler.hpp"

namespace fraudnet {

/// Feed-forward network: dense layers, ReLU hidden activations, a single linear logit output
/// (sigmoid is folded into the binary cross-entropy for numerical stability).
/// With no hidden layers it is exactly logistic regression.
///
/// All parameters live in one flat vector: for each layer, `out*in` weights (row-major,
/// one row per output unit) followed by `out` biases.
class Mlp {
 public:
  Mlp() = default;
  /// `sizes` = {inputs, hidden..., 1}. He-initialised from `seed`.
  Mlp(std::vector<std::size_t> sizes, std::uint64_t seed);

  const std::vector<std::size_t>& sizes() const { return sizes_; }
  std::size_t numParams() const { return params_.size(); }
  std::vector<double>& params() { return params_; }
  const std::vector<double>& params() const { return params_; }

  /// Raw output (log-odds of fraud) for one standardized feature row.
  double logit(const double* x) const;
  double predictProba(const double* x) const;
  std::vector<double> predictProba(const Matrix& x) const;

  /// Mean class-weighted binary cross-entropy over the rows listed in `idx` (+ L2 on weights).
  /// Fills `grad` (resized to numParams()) with the analytic gradient via back-propagation.
  double lossAndGrad(const Matrix& x, const std::vector<int>& y, const std::vector<std::size_t>& idx,
                     double posWeight, double l2, std::vector<double>& grad) const;
  /// Loss only (no gradient).
  double loss(const Matrix& x, const std::vector<int>& y, const std::vector<std::size_t>& idx, double posWeight,
              double l2) const;

  nlohmann::json toJson() const;
  static Mlp fromJson(const nlohmann::json& j);

 private:
  std::vector<std::size_t> sizes_;
  std::vector<std::size_t> offsets_;  // start of each layer's parameter block
  std::vector<double> params_;
  void buildOffsets();
  void forward(const double* x, std::vector<std::vector<double>>& act) const;
};

}  // namespace fraudnet
