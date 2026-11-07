#pragma once
#include <cmath>

namespace fraudnet {

/// Numerically stable logistic function: never evaluates exp() of a large positive argument,
/// so it saturates cleanly to 0 / 1 instead of overflowing.
inline double sigmoid(double z) {
  if (z >= 0) return 1.0 / (1.0 + std::exp(-z));
  const double e = std::exp(z);
  return e / (1.0 + e);
}

/// log(1 + exp(z)) without overflow. Used by the binary cross-entropy computed directly from
/// logits (the "log-sum-exp trick" for two classes), so probabilities are never log()-ed.
inline double softplus(double z) { return z > 0 ? z + std::log1p(std::exp(-z)) : std::log1p(std::exp(z)); }

}  // namespace fraudnet
