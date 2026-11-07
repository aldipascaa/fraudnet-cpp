#include "api.hpp"

#include <chrono>
#include <cmath>
#include <set>

#include "fraudnet/version.hpp"

namespace fraudnet {

using nlohmann::json;

namespace {
const auto kStart = std::chrono::steady_clock::now();

enum class Kind { Number, Integer, Boolean, Category };
struct Field {
  const char* name;
  Kind kind;
  double min, max, def;
  const char* description;
};

const Field kFields[] = {
    {"amount", Kind::Number, 0, 10'000'000, 60, "Transaction amount in currency units"},
    {"hour", Kind::Integer, 0, 23, 14, "Hour of day (0-23) of the transaction"},
    {"merchant_category", Kind::Category, 0, 0, 0, "Merchant category"},
    {"distance_from_home_km", Kind::Number, 0, 20'000, 5, "Distance between transaction and cardholder home (km)"},
    {"distance_from_last_txn_km", Kind::Number, 0, 20'000, 3, "Distance from the previous transaction location (km)"},
    {"txn_count_1h", Kind::Integer, 0, 1000, 0, "Transactions on this card in the previous hour"},
    {"txn_count_24h", Kind::Integer, 0, 10'000, 2, "Transactions on this card in the previous 24 hours"},
    {"avg_amount_30d", Kind::Number, 0, 10'000'000, 60, "Cardholder's average transaction amount over 30 days"},
    {"card_present", Kind::Boolean, 0, 1, 1, "Physical card present (true) or card-not-present (false)"},
    {"is_foreign", Kind::Boolean, 0, 1, 0, "Merchant located in a foreign country"},
    {"account_age_days", Kind::Number, 0, 36'500, 700, "Age of the account in days"},
};

const char* kindName(Kind k) {
  switch (k) {
    case Kind::Number: return "number";
    case Kind::Integer: return "integer";
    case Kind::Boolean: return "boolean";
    case Kind::Category: return "category";
  }
  return "";
}

std::string fmt(double v) {
  if (v == std::floor(v) && std::fabs(v) < 1e15) return std::to_string(static_cast<long long>(v));
  return std::to_string(v);
}
}  // namespace

nlohmann::json inputSchema() {
  json arr = json::array();
  for (const auto& f : kFields) {
    json o = {{"name", f.name}, {"type", kindName(f.kind)}, {"description", f.description}};
    if (f.kind == Kind::Category) {
      o["values"] = json::array();
      for (auto c : kCategories) o["values"].push_back(std::string(c));
      o["default"] = "grocery";
    } else if (f.kind == Kind::Boolean) {
      o["default"] = f.def != 0;
    } else {
      o["min"] = f.min;
      o["max"] = f.max;
      o["default"] = f.def;
    }
    arr.push_back(o);
  }
  return arr;
}

ApiResult errorResult(int status, const std::string& code, const std::string& message,
                      const std::vector<std::string>& details) {
  json b = {{"error", code}, {"message", message}};
  if (!details.empty()) b["details"] = details;
  return {status, b};
}

bool parseTransaction(const json& j, Transaction& tx, std::vector<std::string>& errors, const std::string& prefix) {
  const std::size_t before = errors.size();
  auto err = [&](const std::string& field, const std::string& msg) { errors.push_back(prefix + field + ": " + msg); };
  if (!j.is_object()) {
    errors.push_back(prefix.empty() ? "transaction must be a JSON object" : prefix + "must be a JSON object");
    return false;
  }
  std::set<std::string> known;
  for (const auto& f : kFields) known.insert(f.name);
  for (auto it = j.begin(); it != j.end(); ++it)
    if (!known.count(it.key())) err(it.key(), "unknown field");

  for (const auto& f : kFields) {
    const std::string name = f.name;
    if (!j.contains(name) || j[name].is_null()) {
      err(name, "required field is missing");
      continue;
    }
    const json& v = j[name];
    switch (f.kind) {
      case Kind::Category: {
        if (!v.is_string()) {
          err(name, "must be a string");
          break;
        }
        const std::string s = v.get<std::string>();
        if (categoryIndex(s) < 0) {
          std::string list;
          for (auto c : kCategories) list += (list.empty() ? "" : ", ") + std::string(c);
          err(name, "unknown category '" + s + "' (expected one of: " + list + ")");
          break;
        }
        tx.merchant_category = s;
        break;
      }
      case Kind::Boolean: {
        bool b = false;
        if (v.is_boolean()) b = v.get<bool>();
        else if (v.is_number() && (v.get<double>() == 0.0 || v.get<double>() == 1.0)) b = v.get<double>() == 1.0;
        else {
          err(name, "must be true/false (or 0/1)");
          break;
        }
        if (name == "card_present") tx.card_present = b;
        else tx.is_foreign = b;
        break;
      }
      default: {
        if (!v.is_number()) {
          err(name, "must be a number");
          break;
        }
        const double x = v.get<double>();
        if (!std::isfinite(x)) {
          err(name, "must be finite");
          break;
        }
        if (f.kind == Kind::Integer && x != std::floor(x)) {
          err(name, "must be an integer");
          break;
        }
        if (x < f.min || x > f.max) {
          err(name, "must be between " + fmt(f.min) + " and " + fmt(f.max) + " (got " + fmt(x) + ")");
          break;
        }
        if (name == "amount") tx.amount = x;
        else if (name == "hour") tx.hour = static_cast<int>(x);
        else if (name == "distance_from_home_km") tx.distance_from_home_km = x;
        else if (name == "distance_from_last_txn_km") tx.distance_from_last_txn_km = x;
        else if (name == "txn_count_1h") tx.txn_count_1h = static_cast<int>(x);
        else if (name == "txn_count_24h") tx.txn_count_24h = static_cast<int>(x);
        else if (name == "avg_amount_30d") tx.avg_amount_30d = x;
        else if (name == "account_age_days") tx.account_age_days = x;
      }
    }
  }
  return errors.size() == before;
}

json predictionToJson(const FraudModel& model, const Transaction& tx, std::size_t topK) {
  const Prediction p = model.predict(tx, topK);
  const double thr = model.threshold();
  const char* risk = p.probability >= thr ? "high" : (p.probability >= 0.5 * thr ? "medium" : "low");
  json top = json::array();
  for (const auto& c : p.contributions)
    top.push_back({{"feature", c.feature},
                   {"value", c.value},
                   {"contribution", c.contribution},
                   {"direction", c.contribution >= 0 ? "increases_risk" : "decreases_risk"}});
  return {{"probability", p.probability},
          {"is_fraud", p.isFraud},
          {"label", p.isFraud ? "fraud" : "legit"},
          {"risk_level", risk},
          {"threshold", thr},
          {"top_features", top}};
}

namespace {
bool parseBody(const std::string& body, json& out, ApiResult& err) {
  if (body.empty()) {
    err = errorResult(400, "empty_body", "Request body must be a JSON document");
    return false;
  }
  try {
    out = json::parse(body);
  } catch (const json::parse_error& e) {
    err = errorResult(400, "invalid_json", std::string("Request body is not valid JSON: ") + e.what());
    return false;
  }
  return true;
}
}  // namespace

ApiResult handlePredict(const FraudModel& model, const std::string& body) {
  json j;
  ApiResult err;
  if (!parseBody(body, j, err)) return err;
  Transaction tx;
  std::vector<std::string> errors;
  if (!parseTransaction(j, tx, errors)) return errorResult(400, "validation_failed", "Invalid transaction", errors);
  return {200, predictionToJson(model, tx, 5)};
}

ApiResult handleBatch(const FraudModel& model, const std::string& body) {
  json j;
  ApiResult err;
  if (!parseBody(body, j, err)) return err;
  const json* list = nullptr;
  if (j.is_array()) list = &j;
  else if (j.is_object() && j.contains("transactions") && j["transactions"].is_array()) list = &j["transactions"];
  if (!list) return errorResult(400, "invalid_request", "Expected {\"transactions\": [...]} or a JSON array");
  if (list->empty()) return errorResult(400, "invalid_request", "transactions must not be empty");
  if (list->size() > kMaxBatch)
    return errorResult(413, "batch_too_large", "At most " + std::to_string(kMaxBatch) + " transactions per request");

  std::vector<Transaction> txs(list->size());
  std::vector<std::string> errors;
  for (std::size_t i = 0; i < list->size(); ++i) {
    parseTransaction((*list)[i], txs[i], errors, "transactions[" + std::to_string(i) + "].");
    if (errors.size() > 50) {
      errors.push_back("... too many errors, stopping");
      break;
    }
  }
  if (!errors.empty()) return errorResult(400, "validation_failed", "Invalid transactions in batch", errors);

  json results = json::array();
  std::size_t flagged = 0;
  for (std::size_t i = 0; i < txs.size(); ++i) {
    json r = predictionToJson(model, txs[i], 3);
    r["index"] = i;
    r.erase("threshold");
    flagged += r["is_fraud"].get<bool>() ? 1 : 0;
    results.push_back(std::move(r));
  }
  return {200, {{"count", txs.size()}, {"flagged", flagged}, {"threshold", model.threshold()}, {"results", results}}};
}

ApiResult handleModelInfo(const FraudModel& model) {
  json b = model.meta();
  b["threshold"] = model.threshold();
  b["features"] = columnNames();
  b["feature_groups"] = groupNames();
  b["network"] = {{"layer_sizes", model.network().sizes()}, {"parameters", model.network().numParams()}};
  b["input_schema"] = inputSchema();
  b["server_version"] = FRAUDNET_VERSION;
  return {200, b};
}

ApiResult handleHealth(const FraudModel& model) {
  const double up = std::chrono::duration<double>(std::chrono::steady_clock::now() - kStart).count();
  return {200, {{"status", "ok"},
                {"version", FRAUDNET_VERSION},
                {"model_version", model.meta().value("version", "unknown")},
                {"uptime_seconds", up}}};
}

}  // namespace fraudnet
