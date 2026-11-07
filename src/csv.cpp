#include "csv.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "util.hpp"

namespace fraudnet {

namespace {
const char* const kCols[] = {"amount",       "hour",          "merchant_category",
                             "distance_from_home_km", "distance_from_last_txn_km", "txn_count_1h",
                             "txn_count_24h", "avg_amount_30d", "card_present",
                             "is_foreign",   "account_age_days", "is_fraud"};

double toNumber(const std::string& s, std::size_t line, const char* col) {
  try {
    std::size_t pos = 0;
    const double v = std::stod(s, &pos);
    if (pos != s.size() || !std::isfinite(v)) throw std::invalid_argument("x");
    return v;
  } catch (const std::exception&) {
    throw std::runtime_error("CSV line " + std::to_string(line) + ": invalid number '" + s + "' in column " + col);
  }
}
bool toBool(const std::string& s, std::size_t line, const char* col) {
  if (s == "1" || s == "true" || s == "True") return true;
  if (s == "0" || s == "false" || s == "False") return false;
  throw std::runtime_error("CSV line " + std::to_string(line) + ": expected 0/1 in column " + col + ", got '" + s + "'");
}
}  // namespace

std::string csvHeader() {
  std::string h;
  for (std::size_t i = 0; i < std::size(kCols); ++i) {
    if (i) h += ',';
    h += kCols[i];
  }
  return h;
}

void writeCsv(const std::string& path, const std::vector<Sample>& samples) {
  namespace fs = std::filesystem;
  const fs::path p(path);
  if (p.has_parent_path()) fs::create_directories(p.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot open for writing: " + path);
  out << csvHeader() << '\n';
  char buf[256];
  for (const auto& s : samples) {
    const auto& t = s.tx;
    std::snprintf(buf, sizeof buf, "%.2f,%d,%s,%.2f,%.2f,%d,%d,%.2f,%d,%d,%.1f,%d\n", t.amount, t.hour,
                  t.merchant_category.c_str(), t.distance_from_home_km, t.distance_from_last_txn_km,
                  t.txn_count_1h, t.txn_count_24h, t.avg_amount_30d, t.card_present ? 1 : 0,
                  t.is_foreign ? 1 : 0, t.account_age_days, s.label);
    out << buf;
  }
}

std::vector<Sample> parseCsv(const std::string& text, bool* hasLabels) {
  std::istringstream in(text);
  std::string line;
  std::size_t lineNo = 0;
  if (!std::getline(in, line)) throw std::runtime_error("CSV is empty");
  ++lineNo;
  std::map<std::string, std::size_t> col;
  const auto header = splitString(trim(line), ',');
  for (std::size_t i = 0; i < header.size(); ++i) col[trim(header[i])] = i;
  for (std::size_t i = 0; i + 1 < std::size(kCols); ++i)
    if (!col.count(kCols[i])) throw std::runtime_error(std::string("CSV is missing column: ") + kCols[i]);
  const bool labeled = col.count("is_fraud") > 0;
  if (hasLabels) *hasLabels = labeled;

  std::vector<Sample> out;
  while (std::getline(in, line)) {
    ++lineNo;
    line = trim(line);
    if (line.empty()) continue;
    const auto f = splitString(line, ',');
    if (f.size() < header.size())
      throw std::runtime_error("CSV line " + std::to_string(lineNo) + ": expected " +
                               std::to_string(header.size()) + " fields, got " + std::to_string(f.size()));
    auto get = [&](const char* name) { return trim(f[col.at(name)]); };
    Sample s;
    Transaction& t = s.tx;
    t.amount = toNumber(get("amount"), lineNo, "amount");
    t.hour = static_cast<int>(toNumber(get("hour"), lineNo, "hour"));
    t.merchant_category = get("merchant_category");
    if (categoryIndex(t.merchant_category) < 0)
      throw std::runtime_error("CSV line " + std::to_string(lineNo) + ": unknown merchant_category '" +
                               t.merchant_category + "'");
    t.distance_from_home_km = toNumber(get("distance_from_home_km"), lineNo, "distance_from_home_km");
    t.distance_from_last_txn_km = toNumber(get("distance_from_last_txn_km"), lineNo, "distance_from_last_txn_km");
    t.txn_count_1h = static_cast<int>(toNumber(get("txn_count_1h"), lineNo, "txn_count_1h"));
    t.txn_count_24h = static_cast<int>(toNumber(get("txn_count_24h"), lineNo, "txn_count_24h"));
    t.avg_amount_30d = toNumber(get("avg_amount_30d"), lineNo, "avg_amount_30d");
    t.card_present = toBool(get("card_present"), lineNo, "card_present");
    t.is_foreign = toBool(get("is_foreign"), lineNo, "is_foreign");
    t.account_age_days = toNumber(get("account_age_days"), lineNo, "account_age_days");
    s.label = labeled ? (toBool(get("is_fraud"), lineNo, "is_fraud") ? 1 : 0) : 0;
    out.push_back(std::move(s));
  }
  return out;
}

std::vector<Sample> readCsv(const std::string& path, bool* hasLabels) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open file: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return parseCsv(ss.str(), hasLabels);
}

}  // namespace fraudnet
