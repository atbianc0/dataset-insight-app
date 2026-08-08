# Security policy

## Supported versions

Security fixes are applied to the latest release and the current `main` branch.
Pre-release snapshots and older releases may not receive patches.

## Report a vulnerability

Please use [GitHub private vulnerability reporting](https://github.com/atbianc0/dataset-insight-app/security/advisories/new).
Do not open a public issue for a suspected vulnerability and do not include real
customer data, credentials, or exploit details in logs or screenshots.

Include the affected version or commit, impact, reproducible steps using
synthetic data, and any suggested mitigation. A maintainer should acknowledge a
report within seven days. Timelines for validation and disclosure depend on
severity and fix complexity.

## Data and secret handling

- DataLens does not need an OpenAI key for deterministic analysis.
- Keep `.env` files and credentials out of Git. Rotate a key immediately if it
  is exposed.
- Uploaded files are processed by the running application. Deployment-platform
  logs, memory, network, and retention remain the operator's responsibility.
- The optional AI assistant sends an aggregate/schema analysis context to the
  configured OpenAI API only after the user enables it. Do not enable external
  processing for data you are not authorized to share.
- CSV downloads escape spreadsheet-formula prefixes, but recipients should
  still treat exported files as untrusted input.

This project is an analysis aid, not a security boundary or a suitable place to
process regulated data without an independent deployment and privacy review.
