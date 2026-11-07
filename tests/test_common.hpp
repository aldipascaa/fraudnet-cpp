#pragma once
#include "generator.hpp"
#include "pipeline.hpp"

namespace fraudnet::fixtures {

/// One small trained pipeline shared by all tests (trained lazily, once).
inline const PipelineResult& sharedPipeline() {
  static const PipelineResult r = [] {
    GeneratorConfig g;
    g.rows = 20000;
    g.seed = 123;
    TrainConfig t;
    t.maxEpochs = 25;
    t.patience = 5;
    t.seed = 7;
    SplitConfig s;
    s.seed = 7;
    return trainPipeline(generateDataset(g), t, s);
  }();
  return r;
}

}  // namespace fraudnet::fixtures
