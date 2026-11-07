#include "model.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fraudnet {

nlohmann::json FraudModel::toJson() const {
  nlohmann::json j = meta_;
  j["format"] = kFormat;
  j["schema_version"] = kSchemaVersion;
  j["features"] = columnNames();
  j["categories"] = nlohmann::json::array();
  for (auto c : kCategories) j["categories"].push_back(std::string(c));
  j["scaler"] = features_.scaler().toJson();
  j["network"] = net_.toJson();
  j["threshold"] = threshold_;
  return j;
}

namespace {
template <class Expected>
void requireSameList(const nlohmann::json& j, const char* key, const Expected& expected, const char* hint) {
  if (!j.contains(key) || !j[key].is_array()) throw ModelError(std::string("model is missing the '") + key + "' list");
  const auto& got = j[key];
  const std::size_t n = std::min<std::size_t>(got.size(), expected.size());
  for (std::size_t i = 0; i < n; ++i) {
    const std::string have = got[i].is_string() ? got[i].get<std::string>() : "<non-string>";
    const std::string want(expected[i]);
    if (have != want)
      throw ModelError(std::string("model ") + key + " mismatch at position " + std::to_string(i) + ": model has '" +
                       have + "', this build expects '" + want + "' (" + hint + ")");
  }
  if (got.size() != expected.size())
    throw ModelError(std::string("model has ") + std::to_string(got.size()) + " " + key + ", this build expects " +
                     std::to_string(expected.size()) + " (" + hint + ")");
}
}  // namespace

FraudModel FraudModel::fromJson(const nlohmann::json& j) {
  try {
    if (!j.is_object()) throw ModelError("model document must be a JSON object");

    // 1. Format / schema version.
    const std::string format = j.value("format", std::string());
    if (format.rfind("fraudnet-model/", 0) == 0)
      throw ModelError("legacy model format '" + format + "' (schema 1) is no longer supported; retrain with `fraudnet train`");
    if (format != kFormat) throw ModelError("not a FraudNet model (format is '" + format + "')");
    if (!j.contains("schema_version") || !j["schema_version"].is_number_integer())
      throw ModelError("model has no integer 'schema_version'");
    const int version = j["schema_version"].get<int>();
    if (version > kSchemaVersion)
      throw ModelError("model schema_version " + std::to_string(version) + " is newer than this build supports (max " +
                       std::to_string(kSchemaVersion) + "); upgrade FraudNet");
    if (version < kSchemaVersion)
      throw ModelError("model schema_version " + std::to_string(version) + " is no longer supported (need " +
                       std::to_string(kSchemaVersion) + "); retrain with `fraudnet train`");

    // 2. Feature contract: names, order and one-hot categories must match this build exactly.
    requireSameList(j, "features", columnNames(), "feature list/order must match the build that trains and serves");
    requireSameList(j, "categories", kCategories, "category order defines the one-hot encoding");

    // 3. Parameters (each throws on bad shapes / non-finite values).
    FraudModel m;
    m.features_ = FeaturePipeline(StandardScaler::fromJson(j.at("scaler")));
    m.net_ = Mlp::fromJson(j.at("network"));
    if (m.features_.scaler().mean().size() != numFeatures())
      throw ModelError("scaler has " + std::to_string(m.features_.scaler().mean().size()) + " columns, expected " +
                       std::to_string(numFeatures()));
    if (m.net_.sizes().front() != numFeatures())
      throw ModelError("network expects " + std::to_string(m.net_.sizes().front()) + " inputs, expected " +
                       std::to_string(numFeatures()));
    m.threshold_ = j.at("threshold").get<double>();
    if (!std::isfinite(m.threshold_) || m.threshold_ <= 0.0 || m.threshold_ >= 1.0)
      throw ModelError("threshold must be within (0, 1)");

    m.meta_ = j;
    for (const char* k : {"format", "schema_version", "features", "categories", "scaler", "network", "threshold"})
      m.meta_.erase(k);
    return m;
  } catch (const ModelError&) {
    throw;
  } catch (const nlohmann::json::exception& e) {
    throw ModelError(std::string("malformed model: ") + e.what());
  } catch (const std::runtime_error& e) {
    throw ModelError(std::string("invalid model: ") + e.what());
  }
}

void FraudModel::save(const std::string& path) const {
  const std::filesystem::path p(path);
  if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot open for writing: " + path);
  out << toJson().dump(1) << '\n';
  if (!out) throw std::runtime_error("failed writing model: " + path);
}

FraudModel FraudModel::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw ModelError("cannot open model file: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(ss.str());
  } catch (const nlohmann::json::exception& e) {
    throw ModelError("model file " + path + " is not valid JSON: " + e.what());
  }
  try {
    return fromJson(j);
  } catch (const ModelError& e) {
    throw ModelError("model file " + path + ": " + e.what());
  }
}

}  // namespace fraudnet
