
decide the bridge
# ── COMPLETED ─────────────────────────────────────────────────────────────────

## ✓ P1 · Cash allocation optimiser — monthly split
## ✓ P2 · AR(1) inflation model
## ✓ P3 · Decumulation MC — 40-year horizon (age 54→94)
## ✓ P4 · SORR visibility in decumulation
## ✓ P5 · Joint probability — P(accumulate AND survive 40yr)
## ✓ P7 · Student-t return distribution (fat tails, df=5)
## ✓ P8 · Stock/bond return correlation (Cholesky, −0.2)
## ✓ P8b · Stock return mean reversion AR(1) (−0.25)
## ✓ P8c · Parameter uncertainty (epistemic prior on μ and σ)
## ✓ P10+P11 · Glide path + allocation sweep optimiser
## ✓ Pension inflation adjustment (mc_decumulation_pension_adjustment_rate)
## ✓ CAGR analysis — Calmar ratio, win rate, rolling filter, correlation NaN
## ✓ Unified pipeline main.py (accum → decum → cagr → agent)
## ✓ AI agent — 11-question structured action plan (Q1–Q11)


# ── IN PROGRESS ───────────────────────────────────────────────────────────────

## ETF PORTFOLIO OPTIMISER  [implementing]
Runs after CAGR section. User specifies a ticker subset in config [etf_optimizer].
Four strategies: Max Sortino, Max Sharpe, Min Variance, Equal Weight.
Output: optimal weights per strategy + combined P50-CAGR + σ-5y.
Recommended strategy: Max Sortino (penalises only downside vol — relevant for SORR).
Feeds directly into P6 bridge.


# ── REMAINING ─────────────────────────────────────────────────────────────────

## P6 · ETF→CAGR bridge
Reads optimizer weights → reconstructs combined portfolio daily return series →
computes P50 rolling 10y CAGR + 5y annualized vol →
auto-patches expected_yearly_growth_stocks_percent + mc_stock_return_std_percent.
Pipeline becomes: cagr → optimizer → bridge → accum → decum → agent.
Config field: selected_strategy = "max_sortino" (or "max_sharpe" / "min_variance").

## P9 · Historical stress scenarios (accumulation)
Run accumulation with real 2000–02, 2008–09, 2022 return sequences.
Data already fetched in CAGR section — reuse prices_eur.
Output: "if you started in the worst year, how far below target do you land?"

## Scenario comparison mode
python main.py --scenarios 54,57,60
Runs full pipeline for each retirement year, prints side-by-side table.
High usability value, low implementation cost (~60 lines).

## P13 · HTML / PDF report export
Generate shareable report from existing terminal output.
Lowest urgency — analysis is the bottleneck, not the format.


# ── DEFERRED ──────────────────────────────────────────────────────────────────

## Withdrawal tax per cost basis
Requires tracking cost basis through 40-year decumulation.
Large complexity for ~2pp accuracy gain. Not worth implementing.

## Decumulation IT vs DE tax split
Config already handles the key difference (12.5% vs 26% bond tax).
Withdrawal rates are pre-tax anyway.

## Income shock scenarios in accumulation
User can set annual_net_saving lower and re-run.
Not worth a dedicated feature.
