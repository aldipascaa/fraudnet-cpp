#pragma once
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "predictor.hpp"

namespace fraudnet {

constexpr std::size_t kMaxBatch = 10000;
constexpr std::size_t kParallelBatchMin = 512;

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

/// Decision threshold a caller asked for with `?threshold=`: the fraud label then uses it instead of
/// the threshold tuned into the model. Must be a number strictly between 0 and 1.
std::optional<double> parseThreshold(const std::string& raw, std::string& error);

nlohmann::json predictionToJson(const Predictor& predictor, const Transaction& tx, std::size_t topK,
                                std::optional<double> threshold = std::nullopt);

ApiResult handlePredict(const Predictor& predictor, const std::string& body,
                        std::optional<double> threshold = std::nullopt);
/// Batches of kParallelBatchMin transactions or more are scored on several threads.
ApiResult handleBatch(const Predictor& predictor, const std::string& body,
                      std::optional<double> threshold = std::nullopt);
ApiResult handleModelInfo(const Predictor& predictor);
ApiResult handleHealth(const Predictor& predictor);
ApiResult errorResult(int status, const std::string& code, const std::string& message,
                      const std::vector<std::string>& details = {});

}  // namespace fraudnet
