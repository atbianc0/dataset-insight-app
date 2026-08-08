# Contributing to DataLens

Thanks for helping make tabular analysis more trustworthy. Contributions should
preserve the core rule: recommend prediction only when the data and validation
evidence justify it.

## Set up a development environment

DataLens supports Python 3.11 and 3.12.

```bash
git clone https://github.com/atbianc0/dataset-insight-app.git
cd dataset-insight-app
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

## Make a change

1. Create a focused branch from the latest `main`.
2. Add a regression test before fixing a defect.
3. Keep transformations inside the fitted training pipeline to avoid leakage.
4. Use association language; do not turn predictive relationships into causal claims.
5. Do not add uploaded data, secrets, model artifacts, or generated reports to Git.
6. If a fixture changes, update all fields in `sample_data/fixtures.json` and
   explain its source and license in the pull request.

## Check the change

```bash
python -m ruff check app.py src tests
python -m pytest -W error
python -m pytest --cov=src --cov-branch --cov-report=term-missing
python -m pip_audit -r requirements-dev.txt
docker build --tag datalens:local .
```

For UI changes, also exercise upload, both built-in examples, insights-only
analysis, prediction, external validation, and every download from a clean
browser session. Never weaken an assertion only to make CI pass.

## Pull requests

Keep a pull request small enough to review, describe user-visible behavior,
list the tests run, and disclose any unresolved risk. CI checks Python 3.11 and
3.12, lint, tests and coverage, dependency vulnerabilities, secrets, and the
container. By contributing, you agree that your software contribution is
licensed under the repository's MIT license. Third-party data remains under its
own license.
