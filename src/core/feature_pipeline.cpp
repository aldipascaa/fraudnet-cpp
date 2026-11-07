#include "feature_pipeline.hpp"

#include <algorithm>
#include <stdexcept>

namespace fraudnet {

Matrix FeaturePipeline::rawFeatures(const std::vector<Sample>& samples) {
  Matrix m(samples.size(), numFeatures());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const auto f = extractFeatures(samples[i].tx);
    std::copy(f.begin(), f.end(), m.row(i));
  }
  return m;
}

FeaturePipeline FeaturePipeline::fit(const std::vector<Sample>& train) {
  StandardScaler s;
  s.fit(rawFeatures(train));
  return FeaturePipeline(std::move(s));
}

std::vector<double> FeaturePipeline::transform(const Transaction& tx) const {
  if (!fitted()) throw std::logic_error("FeaturePipeline: not fitted");
  return scaler_.transform(extractFeatures(tx));
}

Matrix FeaturePipeline::transform(const std::vector<Sample>& samples) const {
  if (!fitted()) throw std::logic_error("FeaturePipeline: not fitted");
  Matrix m(samples.size(), numFeatures());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const auto row = transform(samples[i].tx);  // same code path as serving
    std::copy(row.begin(), row.end(), m.row(i));
  }
  return m;
}

}  // namespace fraudnet
