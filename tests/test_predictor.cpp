#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <thread>

#include "numeric.hpp"
#include "predictor.hpp"
#include "test_common.hpp"

using namespace fraudnet;

TEST(Predictor, BatchMatchesSingleAndProbabilityMatchesPredict) {
  const auto& p = fixtures::testPredictor();
  std::vector<Transaction> txs;
  for (std::size_t i = 0; i < 50; ++i) txs.push_back(fixtures::evalData()[i].tx);
  const auto batch = p.predictBatch(txs, 3);
  ASSERT_EQ(batch.size(), txs.size());
  for (std::size_t i = 0; i < txs.size(); ++i) {
    EXPECT_EQ(batch[i].probability, p.probability(txs[i]));
    EXPECT_EQ(batch[i].probability, p.predict(txs[i], 0).probability);
    EXPECT_EQ(batch[i].contributions.size(), 3u);
    EXPECT_EQ(batch[i].isFraud, batch[i].probability >= p.threshold());
  }
}

TEST(Predictor, RejectsInvalidInputAndNullModel) {
  const auto& p = fixtures::testPredictor();
  Transaction t = fixtures::typicalTx();
  t.amount = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(p.probability(t), std::invalid_argument);
  EXPECT_THROW(p.predict(t), std::invalid_argument);
  EXPECT_THROW(Predictor(std::shared_ptr<const FraudModel>{}), std::invalid_argument);
}

TEST(Predictor, ProbabilitiesStayInUnitIntervalForExtremeButValidInputs) {
  const auto& p = fixtures::testPredictor();
  Transaction t = fixtures::suspiciousTx();
  t.amount = 1e12;
  t.distance_from_home_km = 1e9;
  t.txn_count_1h = 1'000'000;
  const double prob = p.probability(t);
  EXPECT_TRUE(std::isfinite(prob));
  EXPECT_GE(prob, 0.0);
  EXPECT_LE(prob, 1.0);
}

TEST(Numeric, SigmoidAndSoftplusAreStableAtExtremes) {
  EXPECT_DOUBLE_EQ(sigmoid(0), 0.5);
  EXPECT_EQ(sigmoid(1000), 1.0);
  EXPECT_EQ(sigmoid(-1000), 0.0);
  EXPECT_TRUE(std::isfinite(softplus(1000)));
  EXPECT_NEAR(softplus(1000), 1000.0, 1e-9);
  EXPECT_NEAR(softplus(-1000), 0.0, 1e-300);
  EXPECT_NEAR(softplus(0), std::log(2.0), 1e-15);
}

// One Predictor shared by many threads must give exactly the single-threaded answers.
TEST(Predictor, ConcurrentPredictionsOnSharedInstanceAreConsistent) {
  const Predictor shared = fixtures::testPredictor();  // copy shares the same immutable model
  constexpr std::size_t kN = 200;
  std::vector<Transaction> txs;
  for (std::size_t i = 0; i < kN; ++i) txs.push_back(fixtures::evalData()[i * 7].tx);
  std::vector<Prediction> expected;
  for (const auto& t : txs) expected.push_back(shared.predict(t, 5));

  constexpr int kThreads = 8;
  std::vector<int> mismatches(kThreads, 0);
  std::vector<std::thread> pool;
  for (int w = 0; w < kThreads; ++w)
    pool.emplace_back([&, w] {
      for (int rep = 0; rep < 5; ++rep)
        for (std::size_t i = 0; i < kN; ++i) {
          const std::size_t k = (i + static_cast<std::size_t>(w) * 31) % kN;  // different order per thread
          const auto got = shared.predict(txs[k], 5);
          bool same = got.probability == expected[k].probability && got.isFraud == expected[k].isFraud &&
                      got.contributions.size() == expected[k].contributions.size();
          for (std::size_t c = 0; same && c < got.contributions.size(); ++c)
            same = got.contributions[c].feature == expected[k].contributions[c].feature &&
                   got.contributions[c].contribution == expected[k].contributions[c].contribution;
          if (!same) ++mismatches[static_cast<std::size_t>(w)];
        }
    });
  for (auto& t : pool) t.join();
  for (int w = 0; w < kThreads; ++w) EXPECT_EQ(mismatches[static_cast<std::size_t>(w)], 0) << "thread " << w;
}

// Pins the output of the committed test model for fixed inputs. If feature engineering, scaling or
// the forward pass ever changes behaviour (e.g. training/serving drift), this fails loudly and the
// model has to be retrained and tests/data/test_model.json regenerated on purpose.
TEST(Predictor, GoldenProbabilitiesForCommittedTestModel) {
  const auto& p = fixtures::testPredictor();
  EXPECT_NEAR(p.probability(fixtures::typicalTx()), 0.001665569258641193, 1e-9);
  EXPECT_NEAR(p.probability(fixtures::suspiciousTx()), 0.9999731786193623, 1e-9);
}
