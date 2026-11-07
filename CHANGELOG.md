# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-11-08

Architecture review release; see [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md) for findings and decisions.

### Changed
- **Model format is now schema 2** (`format: "fraudnet-model"`, `schema_version: 2`, ordered `features` and
  `categories`). Models are validated strictly on load (`ModelError`); v1 models are rejected with a retrain
  hint. `models/fraudnet.json` was retrained and recommitted (metrics identical to 1.0.0: seeded training).
- **API routes are versioned**: `/api/v1/{health,model,predict,predict/batch}`. The old `/api/*` paths remain as
  deprecated aliases (`Deprecation: true` + `Link` header). The web UI uses `/api/v1`.
- Request limits per endpoint (64 KiB single, 4 MiB batch; HTTP 413) and `Content-Type` check (HTTP 415);
  10 s socket timeouts.
- Source split into `src/core` (no HTTP/CLI), `src/server`, `src/cli`; CMake targets `fraudnet_core` <-
  `fraudnet_server` <- `fraudnet`.
- `serve --ui DIR` is now an override for UI development; `--model` falls back to `$FRAUDNET_MODEL`, upward
  search, then the installed `share/fraudnet/models` directory.

### Added
- Web UI embedded in the binary at build time (pure-CMake generator); `serve --no-ui`.
- `FeaturePipeline`: the single raw -> network-input path used by training and serving.
- Immutable, thread-safe `Predictor` (`shared_ptr<const FraudModel>`) owned by the server.
- NaN/domain guards: transactions are validated in feature extraction, non-finite scores and diverging
  training runs raise errors; model loading rejects non-finite parameters.
- `CMakePresets.json`, `fraudnet_warnings` interface target, `FRAUDNET_BUILD_TESTS` and
  `FRAUDNET_WARNINGS_AS_ERRORS` options, `install()` rules.
- Tests: 63 total (was 40) including schema/feature-order rejection, training/serving parity, golden outputs,
  concurrent prediction consistency, payload limits, embedded-UI serving; suite runs in ~2 s (was ~7 s) using a
  committed tiny test model (`tests/data/test_model.json`).

### Removed
- The build-machine source path (`FRAUDNET_SOURCE_DIR`) compiled into the binary.

## [1.0.0] - 2026-11-07

First stable release.

### Added
- Seeded synthetic transaction generator (~2 % fraud, including a non-linear "card testing" pattern) and CSV I/O.
- From-scratch ML stack: feature engineering, `StandardScaler`, multi-layer perceptron with analytic
  back-propagation, class-weighted BCE, Adam, early stopping on validation PR-AUC, logistic-regression baseline.
- Metrics: precision, recall, F1, accuracy, ROC-AUC, PR-AUC, confusion matrix, ROC/PR curves, F1-optimal
  threshold search (tuned on validation, reported on test).
- JSON model format (`fraudnet-model/1`) holding weights, scaler, threshold, metrics and training metadata;
  trained model committed at `models/fraudnet.json`.
- CLI: `generate`, `train`, `evaluate`, `predict`, `serve`, `--version`, `--help`.
- REST API on 127.0.0.1: `POST /api/predict`, `POST /api/predict/batch`, `GET /api/model`, `GET /api/health`
  with exhaustive input validation and per-feature occlusion explanations.
- Single-page web UI: live probability gauge, explanation bars, model-metrics page (confusion matrix, ROC,
  PR and training curves), batch CSV upload with sortable results, light/dark themes, responsive, accessible.
- 40 GoogleTest cases (gradient check, metrics, scaler, serialization round-trip, API, HTTP, training AUC).
