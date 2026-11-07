#include "generator.hpp"

#include <algorithm>
#include <cmath>

#include "util.hpp"

namespace fraudnet {

namespace {
// Order follows kCategories.
const std::vector<double> kLegitCat{0.22, 0.18, 0.12, 0.04, 0.07, 0.12, 0.08, 0.10, 0.05, 0.02};
const std::vector<double> kFraudCat{0.04, 0.04, 0.05, 0.12, 0.28, 0.08, 0.06, 0.20, 0.08, 0.05};
const double kBaseAmount[] = {45, 35, 50, 400, 250, 80, 40, 25, 120, 500};

int wrapHour(double h) {
  int x = static_cast<int>(std::floor(h)) % 24;
  if (x < 0) x += 24;
  return x;
}
double round2(double v) { return std::round(v * 100.0) / 100.0; }
}  // namespace

std::vector<Sample> generateDataset(const GeneratorConfig& cfg) {
  Rng rng(cfg.seed);
  std::vector<Sample> out;
  out.reserve(cfg.rows);
  // Probability that a fraudulent transaction shows a fraud-like value on a given signal.
  constexpr double kSignal = 0.62;

  for (std::size_t n = 0; n < cfg.rows; ++n) {
    const bool fraud = rng.bernoulli(cfg.fraudRate);
    auto odd = [&]() { return fraud && rng.bernoulli(kSignal); };  // fraud-like on this signal?

    Sample s;
    s.label = fraud ? 1 : 0;
    Transaction& t = s.tx;

    const std::size_t cat = rng.weighted(fraud ? kFraudCat : kLegitCat);
    t.merchant_category = std::string(kCategories[cat]);

    t.avg_amount_30d = round2(std::max(5.0, rng.lognormal(std::log(60.0), 0.6)));
    const double scale = std::sqrt(t.avg_amount_30d / 60.0);
    const double base = kBaseAmount[cat] * scale;
    t.amount = odd() ? base * rng.lognormal(std::log(2.6), 0.6) : base * rng.lognormal(0.0, 0.6);
    t.amount = round2(std::max(0.5, t.amount));

    t.hour = odd() ? wrapHour(rng.uniform(-3.0, 5.0)) : (rng.bernoulli(0.85) ? wrapHour(rng.normal(14, 4.5)) : static_cast<int>(rng.index(24)));

    const bool travel = t.merchant_category == "travel";
    if (odd()) {
      t.distance_from_home_km = rng.lognormal(std::log(250.0), 1.0);
    } else {
      const double u = rng.uniform();
      const double farP = travel ? 0.40 : 0.05;
      t.distance_from_home_km = u < farP ? rng.exponential(200) : (u < 0.30 + farP ? rng.exponential(30) : rng.exponential(5));
    }
    t.distance_from_last_txn_km =
        odd() ? rng.lognormal(std::log(150.0), 1.2) : (rng.bernoulli(0.8) ? rng.exponential(6) : rng.exponential(40));

    t.txn_count_1h = odd() ? rng.poisson(2.5) : rng.poisson(0.4);
    t.txn_count_24h = t.txn_count_1h + (odd() ? rng.poisson(4.0) : rng.poisson(2.0));

    if (t.merchant_category == "online_services") {
      t.card_present = false;
    } else if (t.merchant_category == "atm") {
      t.card_present = true;
    } else {
      t.card_present = odd() ? rng.bernoulli(0.12) : rng.bernoulli(0.88);
    }
    t.is_foreign = odd() ? rng.bernoulli(0.35) : rng.bernoulli(travel ? 0.15 : 0.03);

    t.account_age_days = std::round(10.0 * (odd() ? rng.lognormal(4.0, 1.2) : rng.lognormal(6.3, 1.0))) / 10.0;
    t.account_age_days = std::max(1.0, t.account_age_days);

    // "Card testing" pattern (non-monotonic): tiny online charges in rapid bursts. A linear model
    // sees low amounts as safe, so this is where the network's interactions pay off.
    if (fraud && rng.bernoulli(0.18)) {
      t.merchant_category = "online_services";
      t.amount = round2(std::max(0.5, rng.lognormal(std::log(2.5), 0.5)));
      t.txn_count_1h = 2 + rng.poisson(3.0);
      t.txn_count_24h = t.txn_count_1h + rng.poisson(3.0);
      t.card_present = false;
    }

    // Noise: round distances like real exports.
    t.distance_from_home_km = round2(t.distance_from_home_km);
    t.distance_from_last_txn_km = round2(t.distance_from_last_txn_km);
    out.push_back(std::move(s));
  }
  return out;
}

}  // namespace fraudnet
