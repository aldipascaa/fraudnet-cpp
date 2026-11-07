#include <cmath>
#include <gtest/gtest.h>
#include <limits>

#include "feature_pipeline.hpp"
#include "test_common.hpp"

using namespace fraudnet;

TEST(Features, ColumnLayoutIsStableAndGroupsCoverEveryColumn) {
  const auto& names = columnNames();
  ASSERT_EQ(names.size(), numFeatures());
  EXPECT_EQ(names.front(), "log_amount");
  EXPECT_EQ(names.back(), "log_account_age");
  EXPECT_EQ(columnGroups().size(), names.size());
  for (auto g : columnGroups()) EXPECT_LT(g, groupNames().size());
}

TEST(Features, GoldenValuesForKnownTransaction) {
  Transaction t = fixtures::typicalTx();  // amount 40, hour 13, grocery, ...
  const auto f = extractFeatures(t);
  ASSERT_EQ(f.size(), numFeatures());
  EXPECT_DOUBLE_EQ(f[0], std::log1p(40.0));
  EXPECT_NEAR(f[1], std::sin(2.0 * 3.14159265358979323846 * 13.0 / 24.0), 1e-12);
  EXPECT_DOUBLE_EQ(f[3], 0.0);  // is_night
  EXPECT_DOUBLE_EQ(f[4], 1.0);  // cat_grocery
  EXPECT_DOUBLE_EQ(f[5], 0.0);  // cat_restaurant
  EXPECT_DOUBLE_EQ(f[14], std::log1p(3.0));
  EXPECT_DOUBLE_EQ(f[18], std::log((40.0 + 1.0) / (55.0 + 1.0)));
  t.hour = 2;
  EXPECT_DOUBLE_EQ(extractFeatures(t)[3], 1.0);
}

TEST(Features, InvalidTransactionsAreRejectedInsteadOfProducingNaN) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  auto bad = [](auto mutate) {
    Transaction t = fixtures::typicalTx();
    mutate(t);
    return t;
  };
  EXPECT_THROW(extractFeatures(bad([&](Transaction& t) { t.amount = nan; })), std::invalid_argument);
  EXPECT_THROW(extractFeatures(bad([&](Transaction& t) { t.amount = -5; })), std::invalid_argument);
  EXPECT_THROW(extractFeatures(bad([&](Transaction& t) { t.amount = -1; })), std::invalid_argument);  // log1p(-1) = -inf
  EXPECT_THROW(extractFeatures(bad([&](Transaction& t) { t.distance_from_home_km = inf; })), std::invalid_argument);
  EXPECT_THROW(extractFeatures(bad([&](Transaction& t) { t.avg_amount_30d = nan; })), std::invalid_argument);
  EXPECT_THROW(extractFeatures(bad([&](Transaction& t) { t.account_age_days = -2; })), std::invalid_argument);
  EXPECT_THROW(extractFeatures(bad([&](Transaction& t) { t.hour = 24; })), std::invalid_argument);
  EXPECT_THROW(extractFeatures(bad([&](Transaction& t) { t.txn_count_1h = -1; })), std::invalid_argument);
  EXPECT_THROW(extractFeatures(bad([&](Transaction& t) { t.merchant_category = "boats"; })), std::invalid_argument);
  EXPECT_NO_THROW(extractFeatures(fixtures::typicalTx()));
}

// Training/serving skew guard: the matrix the trainer builds and the row the server builds for the
// same transaction must be bit-identical, and so must the resulting scores.
TEST(FeaturePipeline, TrainingMatrixRowsEqualServingRowsBitForBit) {
  const auto& p = fixtures::testPredictor();
  const auto& pipeline = p.model().features();
  std::vector<Sample> subset(fixtures::evalData().begin(), fixtures::evalData().begin() + 300);
  const Matrix batch = pipeline.transform(subset);
  ASSERT_EQ(batch.rows, subset.size());
  ASSERT_EQ(batch.cols, numFeatures());
  const auto trainingPathScores = p.model().network().predictProba(batch);
  for (std::size_t i = 0; i < subset.size(); ++i) {
    const auto row = pipeline.transform(subset[i].tx);
    for (std::size_t c = 0; c < row.size(); ++c) ASSERT_EQ(batch.row(i)[c], row[c]) << "row " << i << " col " << c;
    EXPECT_EQ(trainingPathScores[i], p.probability(subset[i].tx)) << "row " << i;
  }
}

TEST(FeaturePipeline, FitUsesOnlyGivenSamplesAndStandardizesThem) {
  const auto& data = fixtures::evalData();
  std::vector<Sample> train(data.begin(), data.begin() + 2000);
  const auto fp = FeaturePipeline::fit(train);
  ASSERT_TRUE(fp.fitted());
  const Matrix z = fp.transform(train);
  for (std::size_t c = 0; c < z.cols; ++c) {
    double mean = 0;
    for (std::size_t i = 0; i < z.rows; ++i) mean += z.row(i)[c];
    EXPECT_NEAR(mean / static_cast<double>(z.rows), 0.0, 1e-9) << columnNames()[c];
  }
  EXPECT_THROW(FeaturePipeline().transform(fixtures::typicalTx()), std::logic_error);
}
