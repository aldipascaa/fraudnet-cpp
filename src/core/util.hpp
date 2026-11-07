#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fraudnet {

/// Deterministic RNG (xoshiro256** with hand-written distributions so generated
/// data is identical across standard-library implementations).
class Rng {
 public:
  explicit Rng(std::uint64_t seed);
  double uniform();
  double uniform(double a, double b) { return a + (b - a) * uniform(); }
  double normal();
  double normal(double mean, double sd) { return mean + sd * normal(); }
  double lognormal(double mu, double sigma);
  double exponential(double mean);
  bool bernoulli(double p) { return uniform() < p; }
  int poisson(double lambda);
  std::size_t index(std::size_t n);
  std::size_t weighted(const std::vector<double>& weights);
  template <class T>
  void shuffle(std::vector<T>& v) {
    for (std::size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[index(i)]);
  }

 private:
  std::uint64_t s_[4];
  std::uint64_t next();
  bool hasSpare_ = false;
  double spare_ = 0.0;
};

std::string isoTimestampUtc();
/// Looks for `relative` in cwd, the executable directory and their parents, then in the installed
/// layout (<prefix>/share/fraudnet/). Never uses build-time paths.
std::optional<std::string> findUpward(const std::string& relative);
std::string executableDir();
std::vector<std::string> splitString(const std::string& s, char sep);
std::string trim(const std::string& s);

}  // namespace fraudnet
