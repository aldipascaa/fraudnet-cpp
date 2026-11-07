#include <cmath>
#include <gtest/gtest.h>

#include "scaler.hpp"

using namespace fraudnet;

TEST(Scaler, FitComputesMeanAndPopulationStd) {
  Matrix x(4, 2);
  const double vals[4][2] = {{1, 10}, {2, 10}, {3, 10}, {4, 30}};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 2; ++j) x.row(i)[j] = vals[i][j];
  StandardScaler s;
  s.fit(x);
  EXPECT_DOUBLE_EQ(s.mean()[0], 2.5);
  EXPECT_NEAR(s.stddev()[0], std::sqrt(1.25), 1e-12);
  EXPECT_DOUBLE_EQ(s.mean()[1], 15.0);
}

TEST(Scaler, TransformedDataHasZeroMeanUnitVariance) {
  Matrix x(1000, 3);
  for (std::size_t i = 0; i < x.rows; ++i) {
    x.row(i)[0] = static_cast<double>(i);
    x.row(i)[1] = static_cast<double>(i % 10) * 100.0 - 7;
    x.row(i)[2] = std::sin(static_cast<double>(i)) * 0.001;
  }
  StandardScaler s;
  s.fit(x);
  const Matrix z = s.transform(x);
  for (std::size_t j = 0; j < 3; ++j) {
    double mean = 0, var = 0;
    for (std::size_t i = 0; i < z.rows; ++i) mean += z.row(i)[j];
    mean /= static_cast<double>(z.rows);
    for (std::size_t i = 0; i < z.rows; ++i) var += (z.row(i)[j] - mean) * (z.row(i)[j] - mean);
    var /= static_cast<double>(z.rows);
    EXPECT_NEAR(mean, 0.0, 1e-9);
    EXPECT_NEAR(var, 1.0, 1e-9);
  }
}

TEST(Scaler, ConstantColumnDoesNotDivideByZero) {
  Matrix x(5, 1);
  for (std::size_t i = 0; i < 5; ++i) x.row(i)[0] = 4.0;
  StandardScaler s;
  s.fit(x);
  const auto out = s.transform(std::vector<double>{4.0});
  EXPECT_DOUBLE_EQ(out[0], 0.0);
  EXPECT_TRUE(std::isfinite(s.transform(std::vector<double>{9.0})[0]));
}

TEST(Scaler, JsonRoundTripIsExact) {
  Matrix x(50, 2);
  for (std::size_t i = 0; i < 50; ++i) {
    x.row(i)[0] = std::sqrt(static_cast<double>(i));
    x.row(i)[1] = 1.0 / (1.0 + static_cast<double>(i));
  }
  StandardScaler s;
  s.fit(x);
  const auto restored = StandardScaler::fromJson(nlohmann::json::parse(s.toJson().dump()));
  EXPECT_EQ(s.mean(), restored.mean());
  EXPECT_EQ(s.stddev(), restored.stddev());
}

TEST(Scaler, RejectsMalformedInput) {
  StandardScaler s;
  EXPECT_THROW(s.fit(Matrix{}), std::invalid_argument);
  EXPECT_THROW(StandardScaler::fromJson({{"mean", {1.0, 2.0}}, {"std", {1.0}}}), std::runtime_error);
  EXPECT_THROW(s.transform(std::vector<double>{1.0}), std::invalid_argument);
}
