#pragma once
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "features.hpp"
#include "mlp.hpp"
#include "scaler.hpp"

namespace fraudnet {

struct Contribution {
  std::string feature;
  nlohmann::json value;      // raw input value of this feature
  double contribution = 0;   // change in log-odds attributable to this feature (+ = towards fraud)
};

struct Prediction {
  double probability = 0;
  bool isFraud = false;
  std::vector<Contribution> contributions;  // sorted by |contribution| descending
};

/// Deployable model: scaler + network + decision threshold + metadata (metrics, training info).
class FraudModel {
 public:
  FraudModel() = default;
  FraudModel(StandardScaler scaler, Mlp net, double threshold) : scaler_(std::move(scaler)), net_(std::move(net)), threshold_(threshold) {}

  double threshold() const { return threshold_; }
  void setThreshold(double t) { threshold_ = t; }
  const StandardScaler& scaler() const { return scaler_; }
  const Mlp& network() const { return net_; }
  nlohmann::json& meta() { return meta_; }
  const nlohmann::json& meta() const { return meta_; }

  double predictProba(const Transaction& tx) const;
  /// `topK` = number of explanation entries to return (0 = none, large = all).
  Prediction predict(const Transaction& tx, std::size_t topK = 5) const;

  nlohmann::json toJson() const;
  static FraudModel fromJson(const nlohmann::json& j);
  void save(const std::string& path) const;
  static FraudModel load(const std::string& path);

  static constexpr const char* kFormat = "fraudnet-model/1";

 private:
  StandardScaler scaler_;
  Mlp net_;
  double threshold_ = 0.5;
  nlohmann::json meta_ = nlohmann::json::object();
};

}  // namespace fraudnet
