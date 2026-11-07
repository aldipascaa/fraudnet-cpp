#include <cmath>
#include <gtest/gtest.h>

#include "mlp.hpp"
#include "util.hpp"

using namespace fraudnet;

namespace {
struct Toy {
  Matrix x;
  std::vector<int> y;
  std::vector<std::size_t> idx;
};
Toy makeToy(std::size_t n, std::size_t d, std::uint64_t seed) {
  Rng rng(seed);
  Toy t;
  t.x = Matrix(n, d);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < d; ++j) t.x.row(i)[j] = rng.normal();
    t.y.push_back(rng.bernoulli(0.4) ? 1 : 0);
    t.idx.push_back(i);
  }
  return t;
}

double maxRelativeGradError(Mlp& net, const Toy& t, double posWeight, double l2) {
  std::vector<double> analytic;
  net.lossAndGrad(t.x, t.y, t.idx, posWeight, l2, analytic);
  double worst = 0;
  const double h = 1e-6;
  for (std::size_t k = 0; k < net.numParams(); ++k) {
    const double saved = net.params()[k];
    net.params()[k] = saved + h;
    const double up = net.loss(t.x, t.y, t.idx, posWeight, l2);
    net.params()[k] = saved - h;
    const double down = net.loss(t.x, t.y, t.idx, posWeight, l2);
    net.params()[k] = saved;
    const double numeric = (up - down) / (2 * h);
    const double denom = std::max({std::fabs(numeric), std::fabs(analytic[k]), 1e-8});
    worst = std::max(worst, std::fabs(numeric - analytic[k]) / denom);
  }
  return worst;
}
}  // namespace

TEST(MlpGradient, DeepNetworkMatchesFiniteDifferences) {
  Mlp net({6, 8, 5, 1}, 11);
  const Toy t = makeToy(40, 6, 5);
  EXPECT_LT(maxRelativeGradError(net, t, 3.5, 0.01), 1e-5);
}

TEST(MlpGradient, LogisticRegressionMatchesFiniteDifferences) {
  Mlp net({7, 1}, 3);
  const Toy t = makeToy(30, 7, 9);
  EXPECT_LT(maxRelativeGradError(net, t, 1.0, 0.0), 1e-6);
}

TEST(MlpGradient, SubBatchIndicesAreRespected) {
  Mlp net({4, 3, 1}, 1);
  Toy t = makeToy(20, 4, 2);
  t.idx = {1, 5, 7, 11};
  EXPECT_LT(maxRelativeGradError(net, t, 2.0, 0.001), 1e-5);
}

TEST(Mlp, ProbabilitiesAreValidAndStableForExtremeInputs) {
  Mlp net({3, 4, 1}, 2);
  const double big[3] = {1e6, -1e6, 1e6};
  const double p = net.predictProba(big);
  EXPECT_GE(p, 0.0);
  EXPECT_LE(p, 1.0);
  EXPECT_FALSE(std::isnan(p));
}

TEST(Mlp, RejectsBadArchitecture) {
  EXPECT_THROW(Mlp({4, 3}, 1), std::invalid_argument);
  EXPECT_THROW(Mlp({4}, 1), std::invalid_argument);
}

TEST(Mlp, SameSeedSameWeights) {
  Mlp a({5, 4, 1}, 99), b({5, 4, 1}, 99), c({5, 4, 1}, 100);
  EXPECT_EQ(a.params(), b.params());
  EXPECT_NE(a.params(), c.params());
}
