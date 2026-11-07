#include "scaler.hpp"

#include <cmath>
#include <stdexcept>

namespace fraudnet {

void StandardScaler::fit(const Matrix& x) {
  if (x.rows == 0) throw std::invalid_argument("StandardScaler::fit: empty matrix");
  mean_.assign(x.cols, 0.0);
  std_.assign(x.cols, 0.0);
  const double n = static_cast<double>(x.rows);
  for (std::size_t i = 0; i < x.rows; ++i)
    for (std::size_t j = 0; j < x.cols; ++j) mean_[j] += x.row(i)[j];
  for (auto& m : mean_) m /= n;
  for (std::size_t i = 0; i < x.rows; ++i)
    for (std::size_t j = 0; j < x.cols; ++j) {
      const double d = x.row(i)[j] - mean_[j];
      std_[j] += d * d;
    }
  for (auto& s : std_) {
    s = std::sqrt(s / n);
    if (s < 1e-9) s = 1.0;  // constant column: leave it centred but unscaled
  }
}

void StandardScaler::transformRow(const double* in, double* out) const {
  for (std::size_t j = 0; j < mean_.size(); ++j) out[j] = (in[j] - mean_[j]) / std_[j];
}

Matrix StandardScaler::transform(const Matrix& x) const {
  if (x.cols != mean_.size()) throw std::invalid_argument("StandardScaler::transform: column mismatch");
  Matrix out(x.rows, x.cols);
  for (std::size_t i = 0; i < x.rows; ++i) transformRow(x.row(i), out.row(i));
  return out;
}

std::vector<double> StandardScaler::transform(const std::vector<double>& row) const {
  if (row.size() != mean_.size()) throw std::invalid_argument("StandardScaler::transform: size mismatch");
  std::vector<double> out(row.size());
  transformRow(row.data(), out.data());
  return out;
}

nlohmann::json StandardScaler::toJson() const { return {{"mean", mean_}, {"std", std_}}; }

StandardScaler StandardScaler::fromJson(const nlohmann::json& j) {
  StandardScaler s;
  s.mean_ = j.at("mean").get<std::vector<double>>();
  s.std_ = j.at("std").get<std::vector<double>>();
  if (s.mean_.size() != s.std_.size() || s.mean_.empty())
    throw std::runtime_error("scaler: mean/std size mismatch");
  for (double v : s.mean_)
    if (!std::isfinite(v)) throw std::runtime_error("scaler: non-finite mean");
  for (double v : s.std_)
    if (!(v > 0)) throw std::runtime_error("scaler: non-positive std");
  return s;
}

}  // namespace fraudnet
