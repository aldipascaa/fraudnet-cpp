#pragma once
#include <cstdint>
#include <vector>

#include "features.hpp"

namespace fraudnet {

struct GeneratorConfig {
  std::size_t rows = 120000;
  std::uint64_t seed = 42;
  double fraudRate = 0.02;
};

/// Synthetic credit-card-like transactions. Fraud cases deviate from the cardholder's
/// normal behaviour on a random subset of signals, so the classes overlap realistically.
std::vector<Sample> generateDataset(const GeneratorConfig& cfg);

}  // namespace fraudnet
