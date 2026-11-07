#pragma once
#include <vector>

#include <nlohmann/json.hpp>

namespace fraudnet {

/// Row-major dense matrix.
struct Matrix {
  std::size_t rows = 0, cols = 0;
  std::vector<double> data;
  Matrix() = default;
  Matrix(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * c, 0.0) {}
  double* row(std::size_t i) { return data.data() + i * cols; }
  const double* row(std::size_t i) const { return data.data() + i * cols; }
};

/// Feature standardization (zero mean, unit variance), fitted on training data only.
class StandardScaler {
 public:
  void fit(const Matrix& x);
  void transformRow(const double* in, double* out) const;
  Matrix transform(const Matrix& x) const;
  std::vector<double> transform(const std::vector<double>& row) const;
  const std::vector<double>& mean() const { return mean_; }
  const std::vector<double>& stddev() const { return std_; }
  bool fitted() const { return !mean_.empty(); }

  nlohmann::json toJson() const;
  static StandardScaler fromJson(const nlohmann::json& j);

 private:
  std::vector<double> mean_, std_;
};

}  // namespace fraudnet
