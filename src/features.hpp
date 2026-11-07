#pragma once
#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace fraudnet {

inline constexpr std::array<std::string_view, 10> kCategories{
    "grocery", "restaurant", "fuel",     "travel",          "electronics",
    "clothing", "entertainment", "online_services", "atm", "jewelry"};

/// Raw (human-facing) transaction record.
struct Transaction {
  double amount = 0;
  int hour = 12;  // 0..23
  std::string merchant_category = "grocery";
  double distance_from_home_km = 0;
  double distance_from_last_txn_km = 0;
  int txn_count_1h = 0;   // previous transactions on this card in the last hour
  int txn_count_24h = 0;  // previous transactions on this card in the last 24 hours
  double avg_amount_30d = 50;
  bool card_present = true;
  bool is_foreign = false;
  double account_age_days = 365;
};

struct Sample {
  Transaction tx;
  int label = 0;  // 1 = fraud
};

/// Index of a category name, or -1.
int categoryIndex(std::string_view name);

/// Number of engineered numeric columns fed to the model.
std::size_t numFeatures();
/// Engineered feature vector for one transaction.
std::vector<double> extractFeatures(const Transaction& tx);
/// Name of each engineered column.
const std::vector<std::string>& columnNames();
/// For each column, the index of the human-facing feature group it belongs to.
const std::vector<std::size_t>& columnGroups();
/// Human-facing feature groups (one per raw input field), used for explanations.
const std::vector<std::string>& groupNames();
/// Raw value of a group for display.
nlohmann::json groupRawValue(const Transaction& tx, std::size_t group);

}  // namespace fraudnet
