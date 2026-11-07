#pragma once
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "predictor.hpp"

namespace fraudnet {

constexpr std::size_t kMaxBatch = 10000;

struct ApiResult {
  int status = 200;
  nlohmann::json body;
};

/// Describes every accepted input field (also served to the UI via GET /api/v1/model).
nlohmann::json inputSchema();

/// Validates `j` and fills `tx`. Appends human-readable problems to `errors` (each prefixed with
/// `prefix`) and returns true when the transaction is valid.
bool parseTransaction(const nlohmann::json& j, Transaction& tx, std::vector<std::string>& errors,
                      const std::string& prefix = "");

nlohmann::json predictionToJson(const Predictor& predictor, const Transaction& tx, std::size_t topK);

ApiResult handlePredict(const Predictor& predictor, const std::string& body);
ApiResult handleBatch(const Predictor& predictor, const std::string& body);
ApiResult handleModelInfo(const Predictor& predictor);
ApiResult handleHealth(const Predictor& predictor);
ApiResult errorResult(int status, const std::string& code, const std::string& message,
                      const std::vector<std::string>& details = {});

}  // namespace fraudnet
