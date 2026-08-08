# DataLens benchmark protocol

Release performance is measured with the exact fixtures recorded in
`sample_data/fixtures.json`. The profiling timers include parsing, sanitation,
role inference, and exact table-level counts; reported fixture results are the
median of three runs. The end-to-end timer starts before
the churn workflow decision and ends after complete external scoring, metrics,
drift, and entity-overlap analysis.

The release configuration uses `DATALENS_MODEL_SAMPLE_ROWS=60000`, the Standard
three-fold cross-validation effort, seed 42, a deterministic 20% untouched
holdout, and all 64,374 testing rows. Peak resident memory is captured with
`/usr/bin/time -l`; timings are wall-clock seconds on an otherwise idle machine.

These figures are development-machine evidence, not a hosting-service guarantee.
CI independently repeats functional, fixture, and built-container gates.
