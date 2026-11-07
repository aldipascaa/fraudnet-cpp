#pragma once
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "feature_pipeline.hpp"
#include "mlp.hpp"

namespace fraudnet {

/// Thrown when a model artifact cannot be loaded or fails validation.
class ModelError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Trained, deployable artifact: feature pipeline (scaler) + network + decision threshold +
/// free-form metadata (metrics, training configuration). This is a plain value type; to score
/// transactions wrap it in an immutable `Predictor` (predictor.hpp).
///
/// Artifact format (JSON), `schema_version` 2:
///   format          "fraudnet-model"
///   schema_version  integer, bumped on any incompatible change
///   features        ordered engineered column names (must equal this build's feature list)
///   categories      ordered merchant categories (one-hot order; must equal this build's)
///   scaler          {mean[], std[]}
///   network         {sizes[], layers[{in,out,activation,weights,bias}]}
///   threshold       decision threshold in (0,1)
///   ...             any other keys are kept as metadata (name, version, training, metrics)
class FraudModel {
 public:
  static constexpr const char* kFormat = "fraudnet-model";
  static constexpr int kSchemaVersion = 2;

  FraudModel() = default;
  FraudModel(FeaturePipeline features, Mlp net, double threshold)
      : features_(std::move(features)), net_(std::move(net)), threshold_(threshold) {}

  double threshold() const { return threshold_; }
  const FeaturePipeline& features() const { return features_; }
  const Mlp& network() const { return net_; }
  nlohmann::json& meta() { return meta_; }
  const nlohmann::json& meta() const { return meta_; }

  nlohmann::json toJson() const;
  /// Strictly validates schema version, feature list/order, categories, shapes, finiteness and
  /// threshold; throws ModelError with a descriptive message on any mismatch.
  static FraudModel fromJson(const nlohmann::json& j);
  void save(const std::string& path) const;
  static FraudModel load(const std::string& path);

 private:
  FeaturePipeline features_;
  Mlp net_;
  double threshold_ = 0.5;
  nlohmann::json meta_ = nlohmann::json::object();
};

}  // namespace fraudnet
