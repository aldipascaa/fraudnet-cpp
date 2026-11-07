#include <algorithm>
#include <gtest/gtest.h>
#include <httplib.h>

#include "api.hpp"
#include "server.hpp"
#include "test_common.hpp"

using namespace fraudnet;
using nlohmann::json;

namespace {
json validTx() {
  return {{"amount", 120.5},       {"hour", 14},          {"merchant_category", "grocery"},
          {"distance_from_home_km", 4.0}, {"distance_from_last_txn_km", 2.0}, {"txn_count_1h", 0},
          {"txn_count_24h", 2},    {"avg_amount_30d", 60.0}, {"card_present", true},
          {"is_foreign", false},   {"account_age_days", 700}};
}
bool mentions(const json& details, const std::string& s) {
  for (const auto& d : details)
    if (d.get<std::string>().find(s) != std::string::npos) return true;
  return false;
}
}  // namespace

TEST(Api, PredictReturnsProbabilityLabelAndTopFeatures) {
  const auto r = handlePredict(fixtures::sharedPipeline().model, validTx().dump());
  ASSERT_EQ(r.status, 200);
  EXPECT_GE(r.body["probability"].get<double>(), 0.0);
  EXPECT_LE(r.body["probability"].get<double>(), 1.0);
  EXPECT_TRUE(r.body["label"] == "fraud" || r.body["label"] == "legit");
  ASSERT_TRUE(r.body["top_features"].is_array());
  EXPECT_EQ(r.body["top_features"].size(), 5u);
  EXPECT_TRUE(r.body["top_features"][0].contains("contribution"));
}

TEST(Api, ValidationReportsEveryProblem) {
  json tx = validTx();
  tx.erase("amount");
  tx["hour"] = 25;
  tx["merchant_category"] = "boats";
  tx["txn_count_1h"] = 1.5;
  tx["card_present"] = "yes";
  tx["surprise"] = 1;
  const auto r = handlePredict(fixtures::sharedPipeline().model, tx.dump());
  ASSERT_EQ(r.status, 400);
  EXPECT_EQ(r.body["error"], "validation_failed");
  const auto& d = r.body["details"];
  EXPECT_TRUE(mentions(d, "amount: required"));
  EXPECT_TRUE(mentions(d, "hour: must be between 0 and 23"));
  EXPECT_TRUE(mentions(d, "unknown category 'boats'"));
  EXPECT_TRUE(mentions(d, "txn_count_1h: must be an integer"));
  EXPECT_TRUE(mentions(d, "card_present"));
  EXPECT_TRUE(mentions(d, "surprise: unknown field"));
}

TEST(Api, RejectsMalformedBodies) {
  const auto& m = fixtures::sharedPipeline().model;
  EXPECT_EQ(handlePredict(m, "").status, 400);
  EXPECT_EQ(handlePredict(m, "{not json").body["error"], "invalid_json");
  EXPECT_EQ(handlePredict(m, "[1,2]").status, 400);
  EXPECT_EQ(handlePredict(m, json{{"amount", "12"}}.dump()).status, 400);
}

TEST(Api, BatchScoresAllAndReportsIndexOfBadItem) {
  const auto& m = fixtures::sharedPipeline().model;
  json ok = {{"transactions", json::array({validTx(), validTx(), validTx()})}};
  auto r = handleBatch(m, ok.dump());
  ASSERT_EQ(r.status, 200);
  EXPECT_EQ(r.body["count"], 3);
  EXPECT_EQ(r.body["results"].size(), 3u);
  EXPECT_EQ(r.body["results"][2]["index"], 2);

  json bad = ok;
  bad["transactions"][1]["hour"] = -3;
  r = handleBatch(m, bad.dump());
  ASSERT_EQ(r.status, 400);
  EXPECT_TRUE(mentions(r.body["details"], "transactions[1].hour"));

  EXPECT_EQ(handleBatch(m, R"({"transactions": []})").status, 400);
  EXPECT_EQ(handleBatch(m, R"({"foo": 1})").status, 400);
}

TEST(Api, ModelInfoAndHealth) {
  const auto& m = fixtures::sharedPipeline().model;
  const auto info = handleModelInfo(m);
  EXPECT_EQ(info.status, 200);
  EXPECT_TRUE(info.body["metrics"]["test"].contains("roc_auc"));
  EXPECT_TRUE(info.body.contains("version"));
  EXPECT_EQ(info.body["input_schema"].size(), 11u);
  EXPECT_EQ(handleHealth(m).body["status"], "ok");
}

TEST(ApiServer, ServesEndpointsOverHttpOnLoopback) {
  ApiServer server(fixtures::sharedPipeline().model, "");
  const int port = server.bind("127.0.0.1", 0);
  ASSERT_GT(port, 0);
  server.start();
  {
    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(3);
    auto health = cli.Get("/api/health");
    ASSERT_TRUE(health);
    EXPECT_EQ(health->status, 200);
    EXPECT_EQ(json::parse(health->body)["status"], "ok");

    auto pred = cli.Post("/api/predict", validTx().dump(), "application/json");
    ASSERT_TRUE(pred);
    EXPECT_EQ(pred->status, 200);
    EXPECT_TRUE(json::parse(pred->body).contains("probability"));

    auto bad = cli.Post("/api/predict", R"({"amount": -1})", "application/json");
    ASSERT_TRUE(bad);
    EXPECT_EQ(bad->status, 400);
    EXPECT_EQ(json::parse(bad->body)["error"], "validation_failed");

    auto missing = cli.Get("/api/nope");
    ASSERT_TRUE(missing);
    EXPECT_EQ(missing->status, 404);
    EXPECT_EQ(json::parse(missing->body)["error"], "not_found");
  }
  server.stop();
}
