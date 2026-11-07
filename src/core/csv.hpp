#pragma once
#include <string>
#include <vector>

#include "features.hpp"

namespace fraudnet {

/// Header: amount,hour,merchant_category,distance_from_home_km,distance_from_last_txn_km,
///         txn_count_1h,txn_count_24h,avg_amount_30d,card_present,is_foreign,account_age_days,is_fraud
/// The is_fraud column is optional when reading (label defaults to 0).
void writeCsv(const std::string& path, const std::vector<Sample>& samples);
std::vector<Sample> readCsv(const std::string& path, bool* hasLabels = nullptr);
std::vector<Sample> parseCsv(const std::string& text, bool* hasLabels = nullptr);
std::string csvHeader();

}  // namespace fraudnet
