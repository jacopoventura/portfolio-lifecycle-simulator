Scope: I want to understand which portfolio do I need to reach financial independence within a fixed expected horizon, if I am on track to reach 
it. If I am not in track, what do I need to adjust to reach the target. With which instruments can I build the stock part and what cagr is 
reasonable to expect. I also want to understand the risks involved, including the decumulation part.

# Project TODO

Where the scope is genuinely under-served:


## Done

  2. ~~Top-line verdict~~ — FINAL VERDICT block now shows cross-pillar dashboard (Accumulation / Preservation / CAGR realism with ✓/✗), one-liner verdict with primary gap + primary lever, then detailed levers A/B/C. (2026-05-27)
  4. ~~Fat-tail risk modeling~~ — block bootstrap MC added (`mc_return_method = "bootstrap"`). Circular block bootstrap from 99 years of spliced S&P 500 total returns (SPY 1993+ / GSPC pre-1993 with dividend adjustment). Captures real crash clustering, volatility regimes, fat tails. Mean-shifted to planning CAGR. Bootstrap is the active method in config.toml; "parametric" remains available. (2026-05-27) — Last Gaussian holdout closed 2026-07-15: the CAGR-analysis two-phase lost-decade stress test now block-resamples the era's real annual returns (was Gaussian mean/σ draws), and the take-away table gained a historical DCA anchor row (same lump+monthly cash-flow profile as the DCA MC row) for validation.
## Nice to have

- GUI / web frontend
- Email/Slack notification on drift (Priority 2.2) crossing a threshold
- Plotting (PNG output of §4.3 fan chart, §4.11 preservation matrix as heatmap)

## Removed

- `main_fi_date.py` + `reports/section5.py` + `compute_fi_date_distribution` + `_fi_bar_at_year` + `FIDateRow`/`FIDateDistribution` dataclasses (2026-05-26). The "when can I retire?" question is answered by decum's Years-to-retirement tradeoff table (Max safe WD column vs target spend = de-facto FI-year crossover). Maintaining two implementations of the same question was the coherence cost; the tradeoff table's framing is more actionable (sustainability vs target spend) than the FI-date distribution's framing (first-crossing year against a preservation bar).
