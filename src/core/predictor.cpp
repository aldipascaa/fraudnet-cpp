#include "predictor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "numeric.hpp"

namespace fraudnet {

Predictor::Predictor(FraudModel model) : model_(std::make_shared<const FraudModel>(std::move(model))) {}

Predictor::Predictor(std::shared_ptr<const FraudModel> model) : model_(std::move(model)) {
  if (!model_) throw std::invalid_argument("Predictor: null model");
}

namespace {
double checkedLogit(const Mlp& net, const std::vector<double>& z) {
  const double v = net.logit(z.data());
  if (!std::isfinite(v)) throw std::runtime_error("model produced a non-finite score");
  return v;
}
}  // namespace

double Predictor::probability(const Transaction& tx) const {
  return sigmoid(checkedLogit(model_->network(), model_->features().transform(tx)));
}

Prediction Predictor::predict(const Transaction& tx, std::size_t topK) const {
  const Mlp& net = model_->network();
  const auto z = model_->features().transform(tx);
  const double base = checkedLogit(net, z);
  Prediction p;
  p.probability = sigmoid(base);
  p.isFraud = p.probability >= model_->threshold();
  if (topK == 0) return p;

  const auto& groups = columnGroups();
  const auto& names = groupNames();
  for (std::size_t g = 0; g < names.size(); ++g) {
    auto zz = z;
    for (std::size_t c = 0; c < zz.size(); ++c)
      if (groups[c] == g) zz[c] = 0.0;  // training mean after standardization
    Contribution c;
    c.feature = names[g];
    c.value = groupRawValue(tx, g);
    c.contribution = base - checkedLogit(net, zz);
    p.contributions.push_back(std::move(c));
  }
  std::sort(p.contributions.begin(), p.contributions.end(),
            [](const Contribution& a, const Contribution& b) { return std::fabs(a.contribution) > std::fabs(b.contribution); });
  if (p.contributions.size() > topK) p.contributions.resize(topK);
  return p;
}

std::vector<Prediction> Predictor::predictBatch(const std::vector<Transaction>& txs, std::size_t topK) const {
  std::vector<Prediction> out;
  out.reserve(txs.size());
  for (const auto& tx : txs) out.push_back(predict(tx, topK));
  return out;
}

std::vector<double> scoreSamples(const Predictor& predictor, const std::vector<Sample>& samples) {
  std::vector<double> s;
  s.reserve(samples.size());
  for (const auto& x : samples) s.push_back(predictor.probability(x.tx));
  return s;
}

}  // namespace fraudnet
