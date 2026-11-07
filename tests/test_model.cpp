#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include "model.hpp"
#include "predictor.hpp"
#include "test_common.hpp"

using namespace fraudnet;
using nlohmann::json;

namespace {
json goodModelJson() { return fixtures::testPredictor().model().toJson(); }

// Asserts fromJson throws ModelError whose message contains `fragment`.
void expectRejected(const json& j, const std::string& fragment) {
  try {
    FraudModel::fromJson(j);
    FAIL() << "model was accepted; expected an error mentioning: " << fragment;
  } catch (const ModelError& e) {
    EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << "actual message: " << e.what();
  }
}
}  // namespace

TEST(Serialization, JsonRoundTripGivesIdenticalPredictions) {
  const auto& model = fixtures::testPredictor().model();
  const auto restored = FraudModel::fromJson(json::parse(model.toJson().dump()));
  EXPECT_DOUBLE_EQ(restored.threshold(), model.threshold());
  EXPECT_EQ(restored.network().params(), model.network().params());
  EXPECT_EQ(restored.features().scaler().mean(), model.features().scaler().mean());
  const Predictor a(restored);
  const auto tx = fixtures::suspiciousTx();
  EXPECT_DOUBLE_EQ(a.probability(tx), fixtures::testPredictor().probability(tx));
  EXPECT_EQ(restored.meta().at("metrics").at("test").at("roc_auc"), model.meta().at("metrics").at("test").at("roc_auc"));
}

TEST(Serialization, FileSaveAndLoad) {
  const auto& model = fixtures::testPredictor().model();
  const auto path = (std::filesystem::temp_directory_path() / "fraudnet_test_model.json").string();
  model.save(path);
  const auto loaded = FraudModel::load(path);
  EXPECT_DOUBLE_EQ(Predictor(loaded).probability(fixtures::suspiciousTx()),
                   fixtures::testPredictor().probability(fixtures::suspiciousTx()));
  std::filesystem::remove(path);
}

TEST(Serialization, ArtifactCarriesSchemaVersionFeatureListAndCategories) {
  const auto j = goodModelJson();
  EXPECT_EQ(j.at("format"), "fraudnet-model");
  EXPECT_EQ(j.at("schema_version"), FraudModel::kSchemaVersion);
  EXPECT_EQ(j.at("features").size(), numFeatures());
  EXPECT_EQ(j.at("categories").size(), kCategories.size());
  EXPECT_TRUE(j.contains("scaler") && j.contains("threshold") && j.contains("metrics") && j.contains("training"));
}

TEST(Serialization, CommittedProductionModelLoadsAndMatchesThisBuild) {
  const auto m = FraudModel::load("models/fraudnet.json");
  EXPECT_FALSE(m.meta().at("version").get<std::string>().empty());
  const Predictor p(m);
  EXPECT_GT(p.probability(fixtures::suspiciousTx()), p.threshold());
  EXPECT_LT(p.probability(fixtures::typicalTx()), p.threshold());
}

TEST(SerializationErrors, MissingFileAndGarbage) {
  EXPECT_THROW(FraudModel::load("definitely/not/here.json"), ModelError);
  const auto path = (std::filesystem::temp_directory_path() / "fraudnet_garbage.json").string();
  { std::ofstream(path) << "{ this is not json"; }
  EXPECT_THROW(FraudModel::load(path), ModelError);
  std::filesystem::remove(path);
  expectRejected(json::array(), "must be a JSON object");
}

TEST(SerializationErrors, WrongFormatAndSchemaVersions) {
  auto j = goodModelJson();
  j["format"] = "other";
  expectRejected(j, "not a FraudNet model");

  j = goodModelJson();
  j["format"] = "fraudnet-model/1";
  j.erase("schema_version");
  expectRejected(j, "legacy model format");

  j = goodModelJson();
  j.erase("schema_version");
  expectRejected(j, "schema_version");

  j = goodModelJson();
  j["schema_version"] = FraudModel::kSchemaVersion + 1;
  expectRejected(j, "newer than this build supports");

  j = goodModelJson();
  j["schema_version"] = FraudModel::kSchemaVersion - 1;
  expectRejected(j, "no longer supported");
}

TEST(SerializationErrors, FeatureListMustMatchExactlyIncludingOrder) {
  auto j = goodModelJson();
  std::swap(j["features"][1], j["features"][2]);  // same set, different order
  expectRejected(j, "features mismatch at position 1");

  j = goodModelJson();
  j["features"].push_back("extra");
  expectRejected(j, "this build expects");

  j = goodModelJson();
  j["features"].erase(j["features"].size() - 1);
  expectRejected(j, "features");

  j = goodModelJson();
  j["features"][0] = "amount_raw";
  expectRejected(j, "model has 'amount_raw', this build expects 'log_amount'");

  j = goodModelJson();
  std::swap(j["categories"][0], j["categories"][1]);
  expectRejected(j, "categories mismatch");

  j = goodModelJson();
  j.erase("features");
  expectRejected(j, "missing the 'features' list");
}

TEST(SerializationErrors, BadParameters) {
  auto j = goodModelJson();
  j["network"]["layers"][0]["weights"].erase(0);
  expectRejected(j, "weight shape mismatch");

  j = goodModelJson();
  j["threshold"] = 1.5;
  expectRejected(j, "threshold");

  j = goodModelJson();
  j["scaler"]["std"][0] = 0.0;
  expectRejected(j, "non-positive std");

  j = goodModelJson();
  j["scaler"]["mean"].erase(0);
  j["scaler"]["std"].erase(0);
  expectRejected(j, "scaler has");

  j = goodModelJson();
  j["network"]["layers"][0]["activation"] = "tanh";
  expectRejected(j, "activation");

  j = goodModelJson();
  j["network"]["layers"][0]["weights"][0] = "NaN";
  expectRejected(j, "malformed model");

  j = goodModelJson();
  j.erase("network");
  expectRejected(j, "malformed model");
}

TEST(Explanation, SortedByMagnitudeAndLimitedToTopK) {
  const auto& p = fixtures::testPredictor();
  const auto pred = p.predict(fixtures::suspiciousTx(), 4);
  ASSERT_EQ(pred.contributions.size(), 4u);
  for (std::size_t i = 1; i < pred.contributions.size(); ++i)
    EXPECT_GE(std::fabs(pred.contributions[i - 1].contribution), std::fabs(pred.contributions[i].contribution));
  EXPECT_TRUE(p.predict(fixtures::suspiciousTx(), 0).contributions.empty());
}

TEST(Explanation, SuspiciousTransactionScoresHigherThanTypicalOne) {
  const auto& p = fixtures::testPredictor();
  EXPECT_GT(p.probability(fixtures::suspiciousTx()), 0.5);
  EXPECT_LT(p.probability(fixtures::typicalTx()), p.threshold());
}
