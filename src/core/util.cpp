#include "util.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <numbers>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fraudnet {

namespace {
std::uint64_t splitmix(std::uint64_t& x) {
  std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
}  // namespace

Rng::Rng(std::uint64_t seed) : s_{0, 0, 0, 0} {
  for (auto& s : s_) s = splitmix(seed);
}

std::uint64_t Rng::next() {
  const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
  const std::uint64_t t = s_[1] << 17;
  s_[2] ^= s_[0];
  s_[3] ^= s_[1];
  s_[1] ^= s_[2];
  s_[0] ^= s_[3];
  s_[2] ^= t;
  s_[3] = rotl(s_[3], 45);
  return result;
}

double Rng::uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }

double Rng::normal() {
  if (hasSpare_) {
    hasSpare_ = false;
    return spare_;
  }
  double u = uniform();
  if (u < 1e-300) u = 1e-300;
  const double v = uniform();
  const double r = std::sqrt(-2.0 * std::log(u));
  const double a = 2.0 * std::numbers::pi * v;
  spare_ = r * std::sin(a);
  hasSpare_ = true;
  return r * std::cos(a);
}

double Rng::lognormal(double mu, double sigma) { return std::exp(normal(mu, sigma)); }

double Rng::exponential(double mean) {
  double u = uniform();
  if (u < 1e-300) u = 1e-300;
  return -mean * std::log(u);
}

int Rng::poisson(double lambda) {
  const double limit = std::exp(-lambda);
  double p = 1.0;
  int k = 0;
  do {
    ++k;
    p *= uniform();
  } while (p > limit && k < 1000);
  return k - 1;
}

std::size_t Rng::index(std::size_t n) {
  if (n == 0) return 0;
  const auto i = static_cast<std::size_t>(uniform() * static_cast<double>(n));
  return std::min(i, n - 1);
}

std::size_t Rng::weighted(const std::vector<double>& w) {
  double total = 0;
  for (double x : w) total += x;
  double r = uniform() * total;
  for (std::size_t i = 0; i < w.size(); ++i) {
    r -= w[i];
    if (r < 0) return i;
  }
  return w.size() - 1;
}

std::string isoTimestampUtc() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

std::string executableDir() {
#ifdef _WIN32
  char buf[MAX_PATH];
  const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
  if (n > 0 && n < MAX_PATH) return std::filesystem::path(std::string(buf, n)).parent_path().string();
#endif
  return std::filesystem::current_path().string();
}

std::optional<std::string> findUpward(const std::string& relative) {
  namespace fs = std::filesystem;
  const std::vector<fs::path> starts{fs::current_path(), fs::path(executableDir())};
  for (const auto& start : starts) {
    fs::path p = start;
    for (int depth = 0; depth < 6; ++depth) {
      std::error_code ec;
      if (fs::exists(p / relative, ec)) return (p / relative).string();
      if (!p.has_parent_path() || p.parent_path() == p) break;
      p = p.parent_path();
    }
  }
  // Installed layout: <prefix>/bin/fraudnet(.exe) + <prefix>/share/fraudnet/<relative>
  {
    std::error_code ec;
    const fs::path shared = fs::path(executableDir()).parent_path() / "share" / "fraudnet" / relative;
    if (fs::exists(shared, ec)) return shared.string();
  }
  return std::nullopt;
}

std::vector<std::string> splitString(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::size_t start = 0;
  while (true) {
    const auto pos = s.find(sep, start);
    if (pos == std::string::npos) {
      out.push_back(s.substr(start));
      break;
    }
    out.push_back(s.substr(start, pos - start));
    start = pos + 1;
  }
  return out;
}

std::string trim(const std::string& s) {
  std::size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

}  // namespace fraudnet
