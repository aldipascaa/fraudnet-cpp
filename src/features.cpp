#include "features.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace fraudnet {

namespace {
const std::vector<std::string> kGroups{
    "amount",           "hour",          "merchant_category", "distance_from_home_km",
    "distance_from_last_txn_km", "txn_count_1h", "txn_count_24h",     "amount_vs_30d_avg",
    "card_present",     "is_foreign",    "account_age_days"};

struct Layout {
  std::vector<std::string> names;
  std::vector<std::size_t> groups;
  Layout() {
    auto add = [&](const std::string& n, std::size_t g) {
      names.push_back(n);
      groups.push_back(g);
    };
    add("log_amount", 0);
    add("hour_sin", 1);
    add("hour_cos", 1);
    add("is_night", 1);
    for (auto c : kCategories) add("cat_" + std::string(c), 2);
    add("log_distance_home", 3);
    add("log_distance_last", 4);
    add("txn_count_1h", 5);
    add("txn_count_24h", 6);
    add("log_amount_ratio", 7);
    add("card_present", 8);
    add("is_foreign", 9);
    add("log_account_age", 10);
  }
};
const Layout& layout() {
  static const Layout l;
  return l;
}
}  // namespace

int categoryIndex(std::string_view name) {
  for (std::size_t i = 0; i < kCategories.size(); ++i)
    if (kCategories[i] == name) return static_cast<int>(i);
  return -1;
}

std::size_t numFeatures() { return layout().names.size(); }
const std::vector<std::string>& columnNames() { return layout().names; }
const std::vector<std::size_t>& columnGroups() { return layout().groups; }
const std::vector<std::string>& groupNames() { return kGroups; }

std::vector<double> extractFeatures(const Transaction& tx) {
  std::vector<double> f;
  f.reserve(numFeatures());
  f.push_back(std::log1p(tx.amount));
  const double ang = 2.0 * std::numbers::pi * tx.hour / 24.0;
  f.push_back(std::sin(ang));
  f.push_back(std::cos(ang));
  f.push_back(tx.hour <= 5 ? 1.0 : 0.0);
  const int ci = categoryIndex(tx.merchant_category);
  if (ci < 0) throw std::invalid_argument("unknown merchant_category: " + tx.merchant_category);
  for (int i = 0; i < static_cast<int>(kCategories.size()); ++i) f.push_back(i == ci ? 1.0 : 0.0);
  f.push_back(std::log1p(tx.distance_from_home_km));
  f.push_back(std::log1p(tx.distance_from_last_txn_km));
  f.push_back(static_cast<double>(tx.txn_count_1h));
  f.push_back(static_cast<double>(tx.txn_count_24h));
  f.push_back(std::log((tx.amount + 1.0) / (tx.avg_amount_30d + 1.0)));
  f.push_back(tx.card_present ? 1.0 : 0.0);
  f.push_back(tx.is_foreign ? 1.0 : 0.0);
  f.push_back(std::log1p(tx.account_age_days));
  return f;
}

nlohmann::json groupRawValue(const Transaction& tx, std::size_t g) {
  switch (g) {
    case 0: return tx.amount;
    case 1: return tx.hour;
    case 2: return tx.merchant_category;
    case 3: return tx.distance_from_home_km;
    case 4: return tx.distance_from_last_txn_km;
    case 5: return tx.txn_count_1h;
    case 6: return tx.txn_count_24h;
    case 7: return std::round(100.0 * tx.amount / std::max(tx.avg_amount_30d, 0.01)) / 100.0;  // multiple of the 30-day average
    case 8: return tx.card_present;
    case 9: return tx.is_foreign;
    case 10: return tx.account_age_days;
    default: return nullptr;
  }
}

}  // namespace fraudnet
