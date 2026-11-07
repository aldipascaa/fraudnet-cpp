#include "mlp.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "util.hpp"

namespace fraudnet {

namespace {
double sigmoid(double z) {
  if (z >= 0) return 1.0 / (1.0 + std::exp(-z));
  const double e = std::exp(z);
  return e / (1.0 + e);
}
// log(1 + exp(z)) without overflow
double softplus(double z) { return z > 0 ? z + std::log1p(std::exp(-z)) : std::log1p(std::exp(z)); }
}  // namespace

Mlp::Mlp(std::vector<std::size_t> sizes, std::uint64_t seed) : sizes_(std::move(sizes)) {
  if (sizes_.size() < 2 || sizes_.back() != 1) throw std::invalid_argument("Mlp: sizes must end with 1 output");
  for (auto s : sizes_)
    if (s == 0) throw std::invalid_argument("Mlp: zero-sized layer");
  buildOffsets();
  Rng rng(seed);
  for (std::size_t l = 0; l + 1 < sizes_.size(); ++l) {
    const std::size_t in = sizes_[l], out = sizes_[l + 1];
    const double sd = std::sqrt(2.0 / static_cast<double>(in));
    for (std::size_t k = 0; k < in * out; ++k) params_[offsets_[l] + k] = rng.normal(0.0, sd);
  }
}

void Mlp::buildOffsets() {
  offsets_.clear();
  std::size_t total = 0;
  for (std::size_t l = 0; l + 1 < sizes_.size(); ++l) {
    offsets_.push_back(total);
    total += sizes_[l] * sizes_[l + 1] + sizes_[l + 1];
  }
  params_.assign(total, 0.0);
}

void Mlp::forward(const double* x, std::vector<std::vector<double>>& act) const {
  const std::size_t L = sizes_.size() - 1;
  act.resize(L + 1);
  act[0].assign(x, x + sizes_[0]);
  for (std::size_t l = 0; l < L; ++l) {
    const std::size_t in = sizes_[l], out = sizes_[l + 1];
    const double* W = params_.data() + offsets_[l];
    const double* b = W + in * out;
    act[l + 1].resize(out);
    for (std::size_t o = 0; o < out; ++o) {
      double s = b[o];
      const double* w = W + o * in;
      for (std::size_t i = 0; i < in; ++i) s += w[i] * act[l][i];
      act[l + 1][o] = (l + 1 < L) ? std::max(0.0, s) : s;
    }
  }
}

double Mlp::logit(const double* x) const {
  std::vector<std::vector<double>> act;
  forward(x, act);
  return act.back()[0];
}

double Mlp::predictProba(const double* x) const { return sigmoid(logit(x)); }

std::vector<double> Mlp::predictProba(const Matrix& x) const {
  std::vector<double> out(x.rows);
  std::vector<std::vector<double>> act;
  for (std::size_t i = 0; i < x.rows; ++i) {
    forward(x.row(i), act);
    out[i] = sigmoid(act.back()[0]);
  }
  return out;
}

double Mlp::lossAndGrad(const Matrix& x, const std::vector<int>& y, const std::vector<std::size_t>& idx,
                        double posWeight, double l2, std::vector<double>& grad) const {
  grad.assign(params_.size(), 0.0);
  const std::size_t L = sizes_.size() - 1;
  std::vector<std::vector<double>> act, delta(L + 1);
  double total = 0.0;
  for (std::size_t n : idx) {
    forward(x.row(n), act);
    const double z = act[L][0];
    const bool pos = y[n] == 1;
    total += pos ? posWeight * softplus(-z) : softplus(z);
    const double p = sigmoid(z);
    delta[L].assign(1, pos ? -posWeight * (1.0 - p) : p);
    for (std::size_t l = L; l-- > 0;) {
      const std::size_t in = sizes_[l], out = sizes_[l + 1];
      const double* W = params_.data() + offsets_[l];
      double* gW = grad.data() + offsets_[l];
      double* gb = gW + in * out;
      for (std::size_t o = 0; o < out; ++o) {
        const double d = delta[l + 1][o];
        gb[o] += d;
        double* gw = gW + o * in;
        for (std::size_t i = 0; i < in; ++i) gw[i] += d * act[l][i];
      }
      if (l > 0) {
        delta[l].assign(in, 0.0);
        for (std::size_t o = 0; o < out; ++o) {
          const double d = delta[l + 1][o];
          const double* w = W + o * in;
          for (std::size_t i = 0; i < in; ++i) delta[l][i] += w[i] * d;
        }
        for (std::size_t i = 0; i < in; ++i)
          if (act[l][i] <= 0.0) delta[l][i] = 0.0;  // ReLU derivative
      }
    }
  }
  const double inv = 1.0 / static_cast<double>(std::max<std::size_t>(1, idx.size()));
  total *= inv;
  for (auto& g : grad) g *= inv;
  if (l2 > 0) {
    for (std::size_t l = 0; l < L; ++l) {
      const std::size_t nW = sizes_[l] * sizes_[l + 1];
      for (std::size_t k = 0; k < nW; ++k) {
        const double w = params_[offsets_[l] + k];
        total += 0.5 * l2 * w * w;
        grad[offsets_[l] + k] += l2 * w;
      }
    }
  }
  return total;
}

double Mlp::loss(const Matrix& x, const std::vector<int>& y, const std::vector<std::size_t>& idx, double posWeight,
                 double l2) const {
  std::vector<double> g;
  return lossAndGrad(x, y, idx, posWeight, l2, g);
}

nlohmann::json Mlp::toJson() const {
  nlohmann::json layers = nlohmann::json::array();
  for (std::size_t l = 0; l + 1 < sizes_.size(); ++l) {
    const std::size_t in = sizes_[l], out = sizes_[l + 1];
    const auto begin = params_.begin() + static_cast<std::ptrdiff_t>(offsets_[l]);
    layers.push_back({{"in", in},
                      {"out", out},
                      {"activation", (l + 2 < sizes_.size()) ? "relu" : "linear"},
                      {"weights", std::vector<double>(begin, begin + static_cast<std::ptrdiff_t>(in * out))},
                      {"bias", std::vector<double>(begin + static_cast<std::ptrdiff_t>(in * out),
                                                   begin + static_cast<std::ptrdiff_t>(in * out + out))}});
  }
  return {{"sizes", sizes_}, {"layers", layers}};
}

Mlp Mlp::fromJson(const nlohmann::json& j) {
  Mlp m;
  m.sizes_ = j.at("sizes").get<std::vector<std::size_t>>();
  if (m.sizes_.size() < 2 || m.sizes_.back() != 1) throw std::runtime_error("network: invalid sizes");
  for (auto s : m.sizes_)
    if (s == 0 || s > 100000) throw std::runtime_error("network: invalid layer size");
  m.buildOffsets();
  const auto& layers = j.at("layers");
  if (layers.size() + 1 != m.sizes_.size()) throw std::runtime_error("network: layer count mismatch");
  for (std::size_t l = 0; l + 1 < m.sizes_.size(); ++l) {
    const auto w = layers[l].at("weights").get<std::vector<double>>();
    const auto b = layers[l].at("bias").get<std::vector<double>>();
    const std::size_t in = m.sizes_[l], out = m.sizes_[l + 1];
    if (w.size() != in * out || b.size() != out) throw std::runtime_error("network: weight shape mismatch");
    std::copy(w.begin(), w.end(), m.params_.begin() + static_cast<std::ptrdiff_t>(m.offsets_[l]));
    std::copy(b.begin(), b.end(), m.params_.begin() + static_cast<std::ptrdiff_t>(m.offsets_[l] + in * out));
  }
  return m;
}

}  // namespace fraudnet
