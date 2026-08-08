# Changelog

All notable changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and intends to use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) for published releases.

## [1.0.0] - 2026-08-07

### Added

- Dataset profiling, workflow recommendations, insight-first analysis, and
  guarded classification/regression paths.
- Built-in Netflix and churn examples, with an external churn validation set.
- Schema-aware scoring, drift and overlap checks, and report/CSV exports.
- Explicit dataset provenance and fixture-integrity manifest.
- Python 3.11/3.12 CI, lint, coverage, dependency, secret, and container checks.
- MIT licensing for code and separate third-party dataset notices.

### Changed

- Refined categorical dtype handling, identifier detection, leakage checks,
  validation, and baseline comparisons.
- Limited uploads to CSV, TSV, TXT, and XLSX files within documented size and
  shape boundaries.
- Restricted optional AI context to aggregate/schema metadata and kept it
  advisory and opt-in.

### Fixed

- Compatible integer/float differences no longer block scoring.
- Scored exports preserve original rows and identifiers.
- Large-dataset profiling avoids repeated full-data work.

[1.0.0]: https://github.com/atbianc0/dataset-insight-app/releases/tag/v1.0.0
