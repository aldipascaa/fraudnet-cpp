#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>

#include "model.hpp"
#include "test_common.hpp"

using namespace fraudnet;

namespace {
Transaction sampleTx() {
  Transaction t;
  t.amount = 900;
  t.hour = 3;
  t.merchant_category = "electronics";
  t.distance_from_home_km = 800;
  t.distance_from_last_txn_km = 400;
  t.txn_count_1h = 4;
  t.txn_count_24h = 8;
  t.avg_amount_30d = 45;
  t.card_present = false;
  t.is_foreign = true;
  t.account_age_days = 20;
  return t;
}
}  // namespace

TEST(Serialization, JsonRoundTripGivesIdenticalPredictions) {
  const auto& model = fixtures::sharedPipeline().model;
  const auto text = model.toJson().dump();
  const auto restored = FraudModel::fromJson(nlohmann::json::parse(text));
  EXPECT_DOUBLE_EQ(restored.threshold(), model.threshold());
  EXPECT_EQ(restored.network().params(), model.network().params());
  const Transaction tx = sampleTx();
  EXPECT_DOUBLE_EQ(restored.predictProba(tx), model.predictProba(tx));
  // metadata survives too
  EXPECT_EQ(restored.meta().at("metrics").at("test").at("roc_auc"), model.meta().at("metrics").at("test").at("roc_auc"));
}

TEST(Serialization, FileSaveAndLoad) {
  const auto& model = fixtures::sharedPipeline().model;
  const auto path = (std::filesystem::temp_directory_path() / "fraudnet_test_model.json").string();
  model.save(path);
  const auto loaded = FraudModel::load(path);
  EXPECT_DOUBLE_EQ(loaded.predictProba(sampleTx()), model.predictProba(sampleTx()));
  std::filesystem::remove(path);
}

TEST(Serialization, RejectsCorruptModels) {
  EXPECT_THROW(FraudModel::load("definitely/not/here.json"), std::runtime_error);
  EXPECT_THROW(FraudModel::fromJson(nlohmann::json{{"format", "other"}}), std::runtime_error);
  auto j = fixtures::sharedPipeline().model.toJson();
  j["features"].push_back("extra");
  EXPECT_THROW(FraudModel::fromJson(j), std::runtime_error);
  j = fixtures::sharedPipeline().model.toJson();
  j["network"]["layers"][0]["weights"].erase(0);
  EXPECT_THROW(FraudModel::fromJson(j), std::runtime_error);
}

TEST(Explanation, SortedByMagnitudeAndLimitedToTopK) {
  const auto& model = fixtures::sharedPipeline().model;
  const auto p = model.predict(sampleTx(), 4);
  ASSERT_EQ(p.contributions.size(), 4u);
  for (std::size_t i = 1; i < p.contributions.size(); ++i)
    EXPECT_GE(std::fabs(p.contributions[i - 1].contribution), std::fabs(p.contributions[i].contribution));
  EXPECT_TRUE(model.predict(sampleTx(), 0).contributions.empty());
}

TEST(Explanation, SuspiciousTransactionScoresHigherThanTypicalOne) {
  const auto& model = fixtures::sharedPipeline().model;
  Transaction normal;
  normal.amount = 40;
  normal.hour = 13;
  normal.merchant_category = "grocery";
  normal.distance_from_home_km = 3;
  normal.distance_from_last_txn_km = 2;
  normal.avg_amount_30d = 55;
  normal.account_age_days = 900;
  EXPECT_GT(model.predictProba(sampleTx()), 0.5);
  EXPECT_LT(model.predictProba(normal), model.threshold());
}
