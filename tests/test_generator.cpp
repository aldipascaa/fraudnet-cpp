#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>

#include "csv.hpp"
#include "generator.hpp"

using namespace fraudnet;

TEST(Generator, SameSeedIsDeterministicDifferentSeedIsNot) {
  GeneratorConfig a;
  a.rows = 500;
  a.seed = 1;
  GeneratorConfig b = a;
  b.seed = 2;
  const auto x = generateDataset(a), y = generateDataset(a), z = generateDataset(b);
  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_EQ(x[i].label, y[i].label);
    EXPECT_DOUBLE_EQ(x[i].tx.amount, y[i].tx.amount);
  }
  bool differs = false;
  for (std::size_t i = 0; i < x.size(); ++i) differs |= (x[i].tx.amount != z[i].tx.amount);
  EXPECT_TRUE(differs);
}

TEST(Generator, FraudRateIsAboutTwoPercentAndFieldsAreInRange) {
  GeneratorConfig cfg;
  cfg.rows = 50000;
  const auto d = generateDataset(cfg);
  double fraud = 0;
  for (const auto& s : d) {
    fraud += s.label;
    EXPECT_GT(s.tx.amount, 0);
    EXPECT_GE(s.tx.hour, 0);
    EXPECT_LE(s.tx.hour, 23);
    EXPECT_GE(s.tx.txn_count_24h, s.tx.txn_count_1h);
    EXPECT_GE(categoryIndex(s.tx.merchant_category), 0);
    EXPECT_GE(s.tx.distance_from_home_km, 0);
  }
  EXPECT_NEAR(fraud / static_cast<double>(d.size()), 0.02, 0.004);
}

TEST(Csv, WriteReadRoundTrip) {
  GeneratorConfig cfg;
  cfg.rows = 300;
  cfg.fraudRate = 0.2;
  const auto d = generateDataset(cfg);
  const auto path = (std::filesystem::temp_directory_path() / "fraudnet_test.csv").string();
  writeCsv(path, d);
  bool labeled = false;
  const auto r = readCsv(path, &labeled);
  std::filesystem::remove(path);
  EXPECT_TRUE(labeled);
  ASSERT_EQ(r.size(), d.size());
  for (std::size_t i = 0; i < d.size(); ++i) {
    EXPECT_EQ(r[i].label, d[i].label);
    EXPECT_EQ(r[i].tx.merchant_category, d[i].tx.merchant_category);
    EXPECT_NEAR(r[i].tx.amount, d[i].tx.amount, 0.006);
  }
}

TEST(Csv, ReportsHelpfulErrors) {
  EXPECT_THROW(parseCsv(""), std::runtime_error);
  EXPECT_THROW(parseCsv("amount,hour\n1,2\n"), std::runtime_error);
  const std::string head = "amount,hour,merchant_category,distance_from_home_km,distance_from_last_txn_km,txn_count_1h,"
                           "txn_count_24h,avg_amount_30d,card_present,is_foreign,account_age_days\n";
  EXPECT_THROW(parseCsv(head + "abc,1,grocery,1,1,0,0,50,1,0,100\n"), std::runtime_error);
  EXPECT_THROW(parseCsv(head + "5,1,spaceships,1,1,0,0,50,1,0,100\n"), std::runtime_error);
  bool labeled = true;
  const auto ok = parseCsv(head + "5,1,grocery,1,1,0,0,50,1,0,100\n", &labeled);
  EXPECT_FALSE(labeled);
  EXPECT_EQ(ok.size(), 1u);
}
