#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <fstream>

#include "api.hpp"
#include "csv.hpp"
#include "fraudnet/version.hpp"
#include "generator.hpp"
#include "pipeline.hpp"
#include "predictor.hpp"
#include "server.hpp"
#include "util.hpp"

using namespace fraudnet;
using nlohmann::json;

namespace {

struct Args {
  std::string command;
  std::map<std::string, std::string> opts;
  std::set<std::string> flags;

  static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
      std::string s = argv[i];
      if (s.rfind("--", 0) == 0) {
        std::string key = s.substr(2);
        const auto eq = key.find('=');
        if (eq != std::string::npos) {
          a.opts[key.substr(0, eq)] = key.substr(eq + 1);
        } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
          a.opts[key] = argv[++i];
        } else {
          a.flags.insert(key);
        }
      } else if (a.command.empty()) {
        a.command = s;
      } else {
        throw std::runtime_error("unexpected argument: " + s);
      }
    }
    return a;
  }
  bool has(const std::string& k) const { return opts.count(k) || flags.count(k); }
  std::string str(const std::string& k, const std::string& def) const {
    auto it = opts.find(k);
    return it == opts.end() ? def : it->second;
  }
  double num(const std::string& k, double def) const {
    auto it = opts.find(k);
    if (it == opts.end()) return def;
    try {
      std::size_t pos = 0;
      const double v = std::stod(it->second, &pos);
      if (pos != it->second.size()) throw std::invalid_argument("x");
      return v;
    } catch (const std::exception&) {
      throw std::runtime_error("--" + k + " expects a number, got '" + it->second + "'");
    }
  }
  std::size_t count(const std::string& k, std::size_t def) const {
    const double v = num(k, static_cast<double>(def));
    if (v < 0 || v != static_cast<double>(static_cast<std::size_t>(v))) throw std::runtime_error("--" + k + " expects a non-negative integer");
    return static_cast<std::size_t>(v);
  }
};

void usage() {
  std::puts(
      "FraudNet " FRAUDNET_VERSION " - fraud detection neural network written from scratch in C++\n"
      "\n"
      "Usage: fraudnet <command> [options]\n"
      "\n"
      "Commands:\n"
      "  generate   Create a synthetic transactions CSV\n"
      "               --rows N (120000)  --seed S (42)  --fraud-rate R (0.02)  --out FILE (data/transactions.csv)\n"
      "  train      Train logistic-regression baseline + neural network, save model JSON\n"
      "               --data FILE (data/transactions.csv)  --out FILE (models/fraudnet.json)\n"
      "               --hidden 32,16  --lr 0.003  --batch 256  --epochs 60  --patience 8\n"
      "               --class-weight W (default: sqrt(neg/pos))  --l2 1e-5  --seed 42\n"
      "  evaluate   Report metrics of a model on a labeled CSV\n"
      "               --model FILE  --data FILE  [--split test|all (all)]  [--threshold T]\n"
      "  predict    Score transactions\n"
      "               --model FILE  (--json '{...}' | --input FILE.json | --csv FILE)\n"
      "  serve      Start REST API + web UI on 127.0.0.1\n"
      "               --model FILE (models/fraudnet.json)  --port 8080  --ui DIR\n"
      "\n"
      "  --version  Print version      --help  Print this help");
}

std::string resolveModel(const Args& a) {
  const std::string given = a.str("model", "");
  if (!given.empty()) return given;
  if (const char* env = std::getenv("FRAUDNET_MODEL"); env && *env) return env;
  if (auto p = findUpward("models/fraudnet.json")) return *p;
  throw std::runtime_error("no model found; pass --model FILE or run `fraudnet train` first");
}

void printMetrics(const char* title, const Metrics& m) {
  std::printf("%s (n=%zu, fraud=%zu)\n", title, m.n, m.positives);
  std::printf("  threshold  %.4f\n  precision  %.4f\n  recall     %.4f\n  F1         %.4f\n  accuracy   %.4f\n"
              "  ROC-AUC    %.4f\n  PR-AUC     %.4f\n",
              m.threshold, m.precision, m.recall, m.f1, m.accuracy, m.rocAuc, m.prAuc);
  std::printf("  confusion matrix:        pred legit   pred fraud\n"
              "                actual legit  %9zu  %11zu\n"
              "                actual fraud  %9zu  %11zu\n",
              m.cm.tn, m.cm.fp, m.cm.fn, m.cm.tp);
}

int cmdGenerate(const Args& a) {
  GeneratorConfig cfg;
  cfg.rows = a.count("rows", cfg.rows);
  cfg.seed = static_cast<std::uint64_t>(a.num("seed", 42));
  cfg.fraudRate = a.num("fraud-rate", cfg.fraudRate);
  if (cfg.rows == 0) throw std::runtime_error("--rows must be positive");
  if (cfg.fraudRate < 0 || cfg.fraudRate > 1) throw std::runtime_error("--fraud-rate must be within [0, 1]");
  const std::string out = a.str("out", "data/transactions.csv");
  const auto data = generateDataset(cfg);
  writeCsv(out, data);
  std::size_t fraud = 0;
  for (const auto& s : data) fraud += static_cast<std::size_t>(s.label);
  std::printf("Wrote %zu transactions to %s (%zu fraud, %.2f%%), seed %llu\n", data.size(), out.c_str(), fraud,
              100.0 * static_cast<double>(fraud) / static_cast<double>(data.size()),
              static_cast<unsigned long long>(cfg.seed));
  return 0;
}

int cmdTrain(const Args& a) {
  const std::string dataPath = a.str("data", "data/transactions.csv");
  const std::string outPath = a.str("out", "models/fraudnet.json");
  bool labeled = false;
  const auto data = readCsv(dataPath, &labeled);
  if (!labeled) throw std::runtime_error("training data needs an is_fraud column");
  if (data.size() < 100) throw std::runtime_error("need at least 100 rows to train");

  TrainConfig cfg;
  if (a.has("hidden")) {
    cfg.hidden.clear();
    const std::string h = a.str("hidden", "");
    if (h != "0" && !h.empty())
      for (const auto& part : splitString(h, ',')) {
        long v = 0;
        try {
          std::size_t pos = 0;
          v = std::stol(part, &pos);
          if (pos != part.size()) throw std::invalid_argument(part);
        } catch (const std::exception&) {
          throw std::runtime_error("--hidden expects comma-separated integers (e.g. 32,16), got '" + h + "'");
        }
        if (v <= 0 || v > 1024) throw std::runtime_error("--hidden sizes must be in 1..1024");
        cfg.hidden.push_back(static_cast<std::size_t>(v));
      }
  }
  cfg.learningRate = a.num("lr", cfg.learningRate);
  cfg.batchSize = std::max<std::size_t>(1, a.count("batch", cfg.batchSize));
  cfg.maxEpochs = a.count("epochs", cfg.maxEpochs);
  cfg.patience = a.count("patience", cfg.patience);
  cfg.l2 = a.num("l2", cfg.l2);
  cfg.posWeight = a.num("class-weight", -1.0);
  cfg.seed = static_cast<std::uint64_t>(a.num("seed", 42));
  SplitConfig split;
  split.seed = cfg.seed;

  std::printf("Loaded %zu rows from %s\n", data.size(), dataPath.c_str());
  const bool quiet = a.flags.count("quiet") > 0;
  auto res = trainPipeline(data, cfg, split, [&](const std::string& which, const EpochLog& e) {
    if (!quiet)
      std::printf("  [%-6s] epoch %2zu  train loss %.4f  val loss %.4f  val PR-AUC %.4f  val ROC-AUC %.4f\n",
                  which.c_str(), e.epoch, e.trainLoss, e.valLoss, e.valPrAuc, e.valRocAuc);
  });
  std::printf("\nSplit: train %zu / val %zu / test %zu | class weight %.2f | best epoch %zu (of %zu) | %.1fs\n\n",
              res.split.train.size(), res.split.val.size(), res.split.test.size(), res.nnTrain.posWeight,
              res.nnTrain.bestEpoch, res.nnTrain.history.size(), res.nnTrain.seconds);
  printMetrics("Baseline logistic regression - TEST", res.baselineTest);
  std::puts("");
  printMetrics("Neural network - TEST", res.testMetrics);
  res.model.save(outPath);
  std::printf("\nModel saved to %s\n", outPath.c_str());
  return 0;
}

int cmdEvaluate(const Args& a) {
  const Predictor model(FraudModel::load(resolveModel(a)));
  const std::string dataPath = a.str("data", "data/transactions.csv");
  bool labeled = false;
  auto data = readCsv(dataPath, &labeled);
  if (!labeled) throw std::runtime_error("evaluation data needs an is_fraud column");
  const std::string which = a.str("split", "all");
  if (which == "test") {
    const auto& sp = model.model().meta().at("training").at("split");
    SplitConfig cfg;
    cfg.valFraction = sp.at("val_fraction");
    cfg.testFraction = sp.at("test_fraction");
    cfg.seed = sp.at("seed");
    data = stratifiedSplit(data, cfg).test;
  } else if (which != "all") {
    throw std::runtime_error("--split must be 'test' or 'all'");
  }
  const double thr = a.num("threshold", model.threshold());
  const auto scores = scoreSamples(model, data);
  const auto m = evaluate(scores, labelsOf(data), thr);
  printMetrics(("Evaluation on " + which + " rows of " + dataPath).c_str(), m);
  return 0;
}

int cmdPredict(const Args& a) {
  const Predictor model(FraudModel::load(resolveModel(a)));
  if (a.has("json") || a.has("input")) {
    std::string text = a.str("json", "");
    if (a.has("input")) {
      std::ifstream in(a.str("input", ""), std::ios::binary);
      if (!in) throw std::runtime_error("cannot open --input file");
      std::stringstream ss;
      ss << in.rdbuf();
      text = ss.str();
    }
    const auto j = json::parse(text, nullptr, false);
    const bool batch = j.is_array() || (j.is_object() && j.contains("transactions"));
    const auto r = batch ? handleBatch(model, text) : handlePredict(model, text);
    std::printf("%s\n", r.body.dump(2).c_str());
    return r.status == 200 ? 0 : 1;
  }
  if (a.has("csv")) {
    const auto data = readCsv(a.str("csv", ""));
    std::printf("row,probability,label\n");
    for (std::size_t i = 0; i < data.size(); ++i) {
      const double p = model.probability(data[i].tx);
      std::printf("%zu,%.6f,%s\n", i, p, p >= model.threshold() ? "fraud" : "legit");
    }
    return 0;
  }
  throw std::runtime_error("predict needs --json '{...}', --input FILE or --csv FILE");
}

int cmdServe(const Args& a) {
  Predictor predictor(FraudModel::load(resolveModel(a)));
  const double portNum = a.num("port", 8080);
  if (portNum < 0 || portNum > 65535) throw std::runtime_error("--port must be within 0..65535");
  const std::string version = predictor.model().meta().value("version", "?");
  ServerOptions opts;
  opts.serveUi = !a.flags.count("no-ui");
  opts.uiOverrideDir = a.str("ui", "");
  ApiServer server(std::move(predictor), opts);
  const int port = server.bind("127.0.0.1", static_cast<int>(portNum));
  std::printf("FraudNet %s (model %s) listening on http://127.0.0.1:%d\n", FRAUDNET_VERSION, version.c_str(), port);
  if (!opts.serveUi) std::puts("Web UI disabled (--no-ui): serving the API only.");
  else if (opts.uiOverrideDir.empty()) std::puts("Web UI: embedded in the binary");
  else std::printf("Web UI: %s (override)\n", opts.uiOverrideDir.c_str());
  std::puts("API: /api/v1/{health,model,predict,predict/batch}");
  std::puts("Press Ctrl+C to stop.");
  std::fflush(stdout);
  server.run();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a = Args::parse(argc, argv);
    if (a.flags.count("version") || a.command == "version") {
      std::printf("fraudnet %s\n", FRAUDNET_VERSION);
      return 0;
    }
    if (a.command.empty() || a.command == "help" || a.flags.count("help")) {
      usage();
      return a.command.empty() && !a.flags.count("help") ? 2 : 0;
    }
    if (a.command == "generate") return cmdGenerate(a);
    if (a.command == "train") return cmdTrain(a);
    if (a.command == "evaluate") return cmdEvaluate(a);
    if (a.command == "predict") return cmdPredict(a);
    if (a.command == "serve") return cmdServe(a);
    std::fprintf(stderr, "error: unknown command '%s'\n\n", a.command.c_str());
    usage();
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
