Scope: I want to understand which portfolio do I need to reach financial independence within a fixed expected horizon, if I am on track to reach 
it. If I am not in track, what do I need to adjust to reach the target. With which instruments can I build the stock part and what cagr is 
reasonable to expect. I also want to understand the risks involved, including the decumulation part.

# Project TODO

Where the scope is genuinely under-served:


  2. No top-line verdict since PLAN SNAPSHOT was removed. Scope explicitly says "am I on track" — the answer is distributed across section takeaways (each ✗ marker), but
  there's no consolidated "VERDICT: NOT ON TRACK — primary gap is X, primary lever is Y". A reader has to assemble it themselves. Three lines at end of run would close
  this.
  3. Drift tracking (TODO 5.6) — operational gap, not financial. You run this monthly–quarterly. Without delta-vs-last-run, "did my numbers shift?" is invisible. CSV log
  exists but you'd diff manually.
  4. Fat-tail risk modeling (TODO 4.2) is the only modeling gap that matters. Current MC is Gaussian → understates the 2008/2020-style fat tails that are the risk you
  actually care about in decum. The Bad-luck/P5 columns are honest but they're honest about a too-thin distribution.
  5. Policy/pension risk (TODO 4.3) — Italian pension reforms every ~10y, capital gains tax revisions. Not modeled. Real tail risk for your 
     scenario.
## Nice to have

- GUI / web frontend
- Email/Slack notification on drift (Priority 2.2) crossing a threshold
- Plotting (PNG output of §4.3 fan chart, §4.11 preservation matrix as heatmap)

## Removed

- `main_fi_date.py` + `reports/section5.py` + `compute_fi_date_distribution` + `_fi_bar_at_year` + `FIDateRow`/`FIDateDistribution` dataclasses (2026-05-26). The "when can I retire?" question is answered by decum's Years-to-retirement tradeoff table (Max safe WD column vs target spend = de-facto FI-year crossover). Maintaining two implementations of the same question was the coherence cost; the tradeoff table's framing is more actionable (sustainability vs target spend) than the FI-date distribution's framing (first-crossing year against a preservation bar).
