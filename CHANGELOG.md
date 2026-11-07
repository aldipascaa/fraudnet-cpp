# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses [Semantic Versioning](https://semver.org/).

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
