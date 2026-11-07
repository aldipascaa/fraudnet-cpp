#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fraudnet {

namespace {
double sigmoid(double z) {
  if (z >= 0) return 1.0 / (1.0 + std::exp(-z));
  const double e = std::exp(z);
  return e / (1.0 + e);
}
}  // namespace

double FraudModel::predictProba(const Transaction& tx) const {
  const auto z = scaler_.transform(extractFeatures(tx));
  return net_.predictProba(z.data());
}

Prediction FraudModel::predict(const Transaction& tx, std::size_t topK) const {
  Prediction p;
  const auto z = scaler_.transform(extractFeatures(tx));
  const double base = net_.logit(z.data());
  p.probability = sigmoid(base);
  p.isFraud = p.probability >= threshold_;
  if (topK == 0) return p;

  // Occlusion attribution: replace one feature group by its training mean (0 after
  // standardization) and measure how much the log-odds move.
  const auto& groups = columnGroups();
  const auto& names = groupNames();
  for (std::size_t g = 0; g < names.size(); ++g) {
    auto zz = z;
    for (std::size_t c = 0; c < zz.size(); ++c)
      if (groups[c] == g) zz[c] = 0.0;
    Contribution c;
    c.feature = names[g];
    c.value = groupRawValue(tx, g);
    c.contribution = base - net_.logit(zz.data());
    p.contributions.push_back(std::move(c));
  }
  std::sort(p.contributions.begin(), p.contributions.end(),
            [](const Contribution& a, const Contribution& b) { return std::fabs(a.contribution) > std::fabs(b.contribution); });
  if (p.contributions.size() > topK) p.contributions.resize(topK);
  return p;
}

nlohmann::json FraudModel::toJson() const {
  nlohmann::json j = meta_;
  j["format"] = kFormat;
  j["features"] = columnNames();
  j["scaler"] = scaler_.toJson();
  j["network"] = net_.toJson();
  j["threshold"] = threshold_;
  return j;
}

FraudModel FraudModel::fromJson(const nlohmann::json& j) {
  if (!j.is_object() || j.value("format", "") != kFormat) throw std::runtime_error("unsupported model format");
  FraudModel m;
  m.scaler_ = StandardScaler::fromJson(j.at("scaler"));
  m.net_ = Mlp::fromJson(j.at("network"));
  m.threshold_ = j.at("threshold").get<double>();
  const auto features = j.at("features").get<std::vector<std::string>>();
  if (features != columnNames()) throw std::runtime_error("model feature list does not match this build");
  if (m.scaler_.mean().size() != numFeatures() || m.net_.sizes().front() != numFeatures())
    throw std::runtime_error("model input size mismatch");
  m.meta_ = j;
  for (const char* k : {"format", "features", "scaler", "network", "threshold"}) m.meta_.erase(k);
  return m;
}

void FraudModel::save(const std::string& path) const {
  const std::filesystem::path p(path);
  if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot open for writing: " + path);
  out << toJson().dump(1) << '\n';
}

FraudModel FraudModel::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open model file: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  try {
    return fromJson(nlohmann::json::parse(ss.str()));
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error(std::string("invalid model file: ") + e.what());
  }
}

}  // namespace fraudnet
