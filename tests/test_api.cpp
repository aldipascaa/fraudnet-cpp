#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <httplib.h>
#include <sstream>
#include <thread>

#include "api.hpp"
#include "server.hpp"
#include "test_common.hpp"
#include "ui_assets.hpp"

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
std::string readFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Starts a server on an ephemeral loopback port for the lifetime of the fixture object.
struct RunningServer {
  ApiServer server;
  int port;
  explicit RunningServer(ServerOptions o = {}) : server(fixtures::testPredictor(), std::move(o)), port(server.bind("127.0.0.1", 0)) {
    server.start();
  }
  ~RunningServer() { server.stop(); }
};
}  // namespace

TEST(Api, PredictReturnsProbabilityLabelAndTopFeatures) {
  const auto r = handlePredict(fixtures::testPredictor(), validTx().dump());
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
  const auto r = handlePredict(fixtures::testPredictor(), tx.dump());
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
  const auto& m = fixtures::testPredictor();
  EXPECT_EQ(handlePredict(m, "").status, 400);
  EXPECT_EQ(handlePredict(m, "{not json").body["error"], "invalid_json");
  EXPECT_EQ(handlePredict(m, "[1,2]").status, 400);
  EXPECT_EQ(handlePredict(m, json{{"amount", "12"}}.dump()).status, 400);
}

TEST(Api, BatchScoresAllAndReportsIndexOfBadItem) {
  const auto& m = fixtures::testPredictor();
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
  const auto& m = fixtures::testPredictor();
  const auto info = handleModelInfo(m);
  EXPECT_EQ(info.status, 200);
  EXPECT_TRUE(info.body["metrics"]["test"].contains("roc_auc"));
  EXPECT_TRUE(info.body.contains("version"));
  EXPECT_EQ(info.body["schema_version"], FraudModel::kSchemaVersion);
  EXPECT_EQ(info.body["api_version"], "v1");
  EXPECT_EQ(info.body["input_schema"].size(), 11u);
  EXPECT_EQ(handleHealth(m).body["status"], "ok");
}

TEST(ApiServer, ServesVersionedEndpointsOverHttpOnLoopback) {
  RunningServer rs;
  ASSERT_GT(rs.port, 0);
  httplib::Client cli("127.0.0.1", rs.port);
  cli.set_connection_timeout(3);
  auto health = cli.Get("/api/v1/health");
  ASSERT_TRUE(health);
  EXPECT_EQ(health->status, 200);
  EXPECT_EQ(json::parse(health->body)["status"], "ok");
  EXPECT_TRUE(health->get_header_value("Deprecation").empty());

  auto info = cli.Get("/api/v1/model");
  ASSERT_TRUE(info);
  EXPECT_EQ(info->status, 200);

  auto pred = cli.Post("/api/v1/predict", validTx().dump(), "application/json");
  ASSERT_TRUE(pred);
  EXPECT_EQ(pred->status, 200);
  EXPECT_TRUE(json::parse(pred->body).contains("probability"));

  json batch = {{"transactions", json::array({validTx(), validTx()})}};
  auto b = cli.Post("/api/v1/predict/batch", batch.dump(), "application/json; charset=utf-8");
  ASSERT_TRUE(b);
  EXPECT_EQ(b->status, 200);
  EXPECT_EQ(json::parse(b->body)["count"], 2);

  auto bad = cli.Post("/api/v1/predict", R"({"amount": -1})", "application/json");
  ASSERT_TRUE(bad);
  EXPECT_EQ(bad->status, 400);
  EXPECT_EQ(json::parse(bad->body)["error"], "validation_failed");

  auto missing = cli.Get("/api/v1/nope");
  ASSERT_TRUE(missing);
  EXPECT_EQ(missing->status, 404);
  EXPECT_EQ(json::parse(missing->body)["error"], "not_found");
}

TEST(ApiServer, LegacyUnversionedRoutesStillWorkAndAreMarkedDeprecated) {
  RunningServer rs;
  httplib::Client cli("127.0.0.1", rs.port);
  auto pred = cli.Post("/api/predict", validTx().dump(), "application/json");
  ASSERT_TRUE(pred);
  EXPECT_EQ(pred->status, 200);
  EXPECT_EQ(pred->get_header_value("Deprecation"), "true");
  EXPECT_NE(pred->get_header_value("Link").find("/api/v1/predict"), std::string::npos);
  auto health = cli.Get("/api/health");
  ASSERT_TRUE(health);
  EXPECT_EQ(health->status, 200);
}

TEST(ApiServer, EnforcesPayloadLimitsAndContentType) {
  ServerOptions o;
  o.maxBodyBytes = 8 * 1024;
  o.maxSingleBodyBytes = 1024;
  RunningServer rs(o);
  httplib::Client cli("127.0.0.1", rs.port);
  cli.set_connection_timeout(3);

  // Single-transaction endpoint: over its own (smaller) limit -> 413 with JSON error.
  json big = validTx();
  big["padding"] = std::string(2000, 'x');
  auto r1 = cli.Post("/api/v1/predict", big.dump(), "application/json");
  ASSERT_TRUE(r1);
  EXPECT_EQ(r1->status, 413);
  EXPECT_EQ(json::parse(r1->body)["error"], "payload_too_large");

  // Batch endpoint under the global cap still works...
  json small = {{"transactions", json::array({validTx()})}};
  auto r2 = cli.Post("/api/v1/predict/batch", small.dump(), "application/json");
  ASSERT_TRUE(r2);
  EXPECT_EQ(r2->status, 200);

  // ...and over the global cap is rejected before parsing.
  json huge = {{"transactions", json::array()}};
  for (int i = 0; i < 100; ++i) huge["transactions"].push_back(validTx());
  const std::string hugeBody = huge.dump();
  ASSERT_GT(hugeBody.size(), o.maxBodyBytes);
  auto r3 = cli.Post("/api/v1/predict/batch", hugeBody, "application/json");
  ASSERT_TRUE(r3);
  EXPECT_EQ(r3->status, 413);

  auto r4 = cli.Post("/api/v1/predict", validTx().dump(), "text/plain");
  ASSERT_TRUE(r4);
  EXPECT_EQ(r4->status, 415);
  EXPECT_EQ(json::parse(r4->body)["error"], "unsupported_media_type");
}

TEST(ApiServer, ServesEmbeddedUiWithoutAnyFilesOnDisk) {
  // The test deliberately does not point the server at a directory: assets come from the binary.
  RunningServer rs;
  httplib::Client cli("127.0.0.1", rs.port);
  auto index = cli.Get("/");
  ASSERT_TRUE(index);
  EXPECT_EQ(index->status, 200);
  EXPECT_NE(index->get_header_value("Content-Type").find("text/html"), std::string::npos);
  EXPECT_NE(index->body.find("<title>"), std::string::npos);
  EXPECT_EQ(index->get_header_value("X-Content-Type-Options"), "nosniff");

  // Every file in ui/ is embedded byte-for-byte and served with a sensible type.
  std::size_t checked = 0;
  for (const auto& entry : std::filesystem::directory_iterator("ui")) {
    const std::string url = "/" + entry.path().filename().string();
    auto res = cli.Get(url);
    ASSERT_TRUE(res) << url;
    EXPECT_EQ(res->status, 200) << url;
    EXPECT_EQ(res->body, readFile(entry.path().string())) << url << " differs from ui/ source";
    ++checked;
  }
  EXPECT_GE(checked, 4u);
  EXPECT_NE(cli.Get("/app.js")->get_header_value("Content-Type").find("javascript"), std::string::npos);
  EXPECT_NE(cli.Get("/style.css")->get_header_value("Content-Type").find("text/css"), std::string::npos);
  EXPECT_NE(cli.Get("/favicon.svg")->get_header_value("Content-Type").find("image/svg+xml"), std::string::npos);

  EXPECT_EQ(cli.Get("/nope.txt")->status, 404);
  EXPECT_EQ(cli.Get("/../CMakeLists.txt")->status, 404);
  EXPECT_EQ(cli.Get("/%2e%2e/CMakeLists.txt")->status, 404);
}

TEST(ApiServer, UiCanBeDisabledOrOverriddenFromDirectory) {
  {
    ServerOptions o;
    o.serveUi = false;
    RunningServer rs(o);
    httplib::Client cli("127.0.0.1", rs.port);
    EXPECT_EQ(cli.Get("/")->status, 404);
    EXPECT_EQ(cli.Get("/api/v1/health")->status, 200);
  }
  {
    const auto dir = std::filesystem::temp_directory_path() / "fraudnet_ui_override";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir / "index.html") << "<html>override-marker</html>"; }
    ServerOptions o;
    o.uiOverrideDir = dir.string();
    RunningServer rs(o);
    httplib::Client cli("127.0.0.1", rs.port);
    auto res = cli.Get("/");
    ASSERT_TRUE(res);
    EXPECT_NE(res->body.find("override-marker"), std::string::npos);
    std::filesystem::remove_all(dir);
  }
  ServerOptions bad;
  bad.uiOverrideDir = "definitely/not/a/dir";
  EXPECT_THROW(ApiServer(fixtures::testPredictor(), bad), std::runtime_error);
}

TEST(ApiServer, EmbeddedAssetTableIsComplete) {
  std::size_t n = 0;
  const EmbeddedAsset* all = embeddedUiAssets(n);
  ASSERT_GE(n, 4u);
  for (std::size_t i = 0; i < n; ++i) EXPECT_GT(all[i].size, 0u) << all[i].path;
  EXPECT_NE(findEmbeddedAsset("/"), nullptr);
  EXPECT_EQ(findEmbeddedAsset("/"), findEmbeddedAsset("/index.html"));
  EXPECT_EQ(findEmbeddedAsset("/missing"), nullptr);
}

TEST(ApiServer, ConcurrentHttpPredictionsAreConsistent) {
  RunningServer rs;
  const std::string body = validTx().dump();
  double expected = 0;
  {
    httplib::Client cli("127.0.0.1", rs.port);
    expected = json::parse(cli.Post("/api/v1/predict", body, "application/json")->body)["probability"];
  }
  constexpr int kThreads = 8, kReps = 25;
  std::atomic<int> bad{0}, failed{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < kThreads; ++t)
    pool.emplace_back([&] {
      httplib::Client cli("127.0.0.1", rs.port);
      cli.set_connection_timeout(5);
      for (int i = 0; i < kReps; ++i) {
        auto r = cli.Post("/api/v1/predict", body, "application/json");
        if (!r || r->status != 200) {
          ++failed;
          continue;
        }
        if (json::parse(r->body)["probability"].get<double>() != expected) ++bad;
      }
    });
  for (auto& t : pool) t.join();
  EXPECT_EQ(failed.load(), 0);
  EXPECT_EQ(bad.load(), 0);
}

TEST(Api, ThresholdOverrideChangesTheLabelButNotTheScore) {
  const auto& m = fixtures::testPredictor();
  const auto base = handlePredict(m, validTx().dump());
  ASSERT_EQ(base.status, 200);
  EXPECT_EQ(base.body["threshold_source"], "model");
  EXPECT_DOUBLE_EQ(base.body["threshold"].get<double>(), m.threshold());
  const double p = base.body["probability"].get<double>();
  ASSERT_GT(p, 0.0);
  ASSERT_LT(p, 1.0);

  // A threshold just below the score flags it; one just above clears it.
  const auto low = handlePredict(m, validTx().dump(), p * 0.5);
  EXPECT_EQ(low.body["is_fraud"], true);
  EXPECT_EQ(low.body["label"], "fraud");
  EXPECT_EQ(low.body["threshold_source"], "request");
  EXPECT_DOUBLE_EQ(low.body["threshold"].get<double>(), p * 0.5);
  const auto high = handlePredict(m, validTx().dump(), p + (1.0 - p) * 0.5);
  EXPECT_EQ(high.body["is_fraud"], false);
  EXPECT_EQ(high.body["label"], "legit");
  for (const auto* r : {&low, &high}) EXPECT_DOUBLE_EQ(r->body["probability"].get<double>(), p);

  const auto batch = handleBatch(m, json{{"transactions", json::array({validTx(), validTx()})}}.dump(), p * 0.5);
  ASSERT_EQ(batch.status, 200);
  EXPECT_EQ(batch.body["flagged"], 2);
  EXPECT_EQ(batch.body["threshold_source"], "request");
}

TEST(Api, ParseThresholdAcceptsOnlyTheOpenUnitInterval) {
  std::string error;
  EXPECT_DOUBLE_EQ(parseThreshold("0.25", error).value(), 0.25);
  EXPECT_DOUBLE_EQ(parseThreshold("1e-3", error).value(), 0.001);
  for (const char* bad : {"", "0", "1", "-0.1", "1.5", "abc", "0.5x", " 0.5", "nan", "inf"}) {
    error.clear();
    EXPECT_FALSE(parseThreshold(bad, error).has_value()) << bad;
    EXPECT_NE(error.find("threshold"), std::string::npos) << bad;
  }
}

TEST(Api, LargeBatchScoredInParallelMatchesSequentialScoring) {
  const auto& m = fixtures::testPredictor();
  json txs = json::array();
  const char* categories[] = {"grocery", "electronics", "travel"};
  for (std::size_t i = 0; i < kParallelBatchMin * 3 + 7; ++i) {
    json tx = validTx();
    tx["amount"] = 5.0 + static_cast<double>(i % 97) * 31.5;
    tx["hour"] = static_cast<int>(i % 24);
    tx["merchant_category"] = categories[i % 3];
    tx["is_foreign"] = i % 5 == 0;
    txs.push_back(std::move(tx));
  }
  const auto r = handleBatch(m, json{{"transactions", txs}}.dump());
  ASSERT_EQ(r.status, 200);
  ASSERT_EQ(r.body["results"].size(), txs.size());
  std::size_t flagged = 0;
  for (std::size_t i = 0; i < txs.size(); ++i) {
    const auto& got = r.body["results"][i];
    const auto one = handlePredict(m, txs[i].dump());
    ASSERT_EQ(got["index"], i);
    ASSERT_DOUBLE_EQ(got["probability"].get<double>(), one.body["probability"].get<double>()) << "row " << i;
    ASSERT_EQ(got["is_fraud"], one.body["is_fraud"]);
    flagged += got["is_fraud"].get<bool>() ? 1 : 0;
  }
  EXPECT_EQ(r.body["flagged"], flagged);
}

TEST(ApiServer, ThresholdQueryParameterIsAppliedAndValidated) {
  RunningServer rs;
  httplib::Client cli("127.0.0.1", rs.port);
  auto res = cli.Post("/api/v1/predict?threshold=0.000001", validTx().dump(), "application/json");
  ASSERT_TRUE(res);
  ASSERT_EQ(res->status, 200);
  const json body = json::parse(res->body);
  EXPECT_EQ(body["threshold_source"], "request");
  EXPECT_EQ(body["is_fraud"], body["probability"].get<double>() >= 0.000001);

  res = cli.Post("/api/v1/predict/batch?threshold=0.5", json{{"transactions", json::array({validTx()})}}.dump(), "application/json");
  ASSERT_TRUE(res);
  EXPECT_EQ(json::parse(res->body)["threshold"], 0.5);

  for (const char* bad : {"/api/v1/predict?threshold=2", "/api/v1/predict?threshold=", "/api/v1/predict/batch?threshold=high"}) {
    res = cli.Post(bad, validTx().dump(), "application/json");
    ASSERT_TRUE(res) << bad;
    EXPECT_EQ(res->status, 400) << bad;
    EXPECT_EQ(json::parse(res->body)["error"], "invalid_threshold") << bad;
  }
}
