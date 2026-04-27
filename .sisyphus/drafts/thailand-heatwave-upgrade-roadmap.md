# Draft: Thailand Heatwave Upgrade Roadmap

## Requirements (confirmed)
- [user intent]: Find detailed research papers and best practices, then determine what the current Thailand heatwave forecasting backend should change.
- [deliverable]: Produce a plan before implementation.
- [repo scope]: Use the existing ERA5 ingestion, ConvLSTM/RandomForest training paths, and Flask API as the baseline.

## Technical Decisions
- [planning mode]: Produce a repo-specific implementation plan, not code changes.
- [scope focus]: Prioritize data variables, targets, evaluation, loss/objective, architecture, and API output implications.
- [test strategy]: No existing test suite; verification will use agent-executed training/evaluation commands plus API smoke checks.
- [canonical model path]: Treat ConvLSTM as the primary training/inference model; keep RandomForest only as a baseline comparator.
- [compatibility]: Preserve existing API fields and add calibrated heat-risk data additively instead of replacing current response keys.

## Research Findings
- [Nature/npj 2024]: 2023 Southeast Asia heatwave was driven by moisture deficiency, tropical waves/MJO, and strong land-atmosphere coupling.
- [Science of the Total Environment 2020]: Southeast Asia heatwaves intensify more strongly when humidity is considered; warm-night and wet-bulb effects matter.
- [Weather ML benchmarks]: Modern weather ML emphasizes multivariate inputs, probabilistic outputs, and strict benchmark/evaluation protocols.
- [ERA5 documentation]: ERA5 provides hourly single-level and pressure-level variables plus ensemble spread useful for uncertainty handling.
- [Metis guardrail]: Compute climatology/percentile thresholds from training-year blocks only; do not allow random splits to headline model quality.
- [Metis guardrail]: Add new ERA5 variables behind explicit configuration rather than ingesting every available field.

## Open Questions
- None blocking for plan generation; proceed with percentile-based event definition and tests-after verification as default assumptions.

## Scope Boundaries
- INCLUDE: data pipeline upgrades, training/evaluation upgrades, model/loss upgrades, API contract upgrades for heat-risk outputs, verification workflow.
- EXCLUDE: immediate code implementation, infra/cloud redesign, unrelated UI work, new external data procurement beyond optional future tasks.
