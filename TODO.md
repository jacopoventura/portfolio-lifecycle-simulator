# Financial Planning Tool — Master Priority List
# Context: early retirement at 54 (2041, 15y horizon), no pensions, 35–40yr decumulation

                                                                                   
  ┌─────┬──────────────────────────────────────┬────────────────────────────────────────────────────────┐
  │  #  │               Feature                │          Why it matters for you specifically           │                                                                      │
  ├─────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────┤                                                               
  │ 4   │ ETF → CAGR bridge                    │ Makes accumulation assumptions evidence-based              │                                                               
  ├─────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────┤                                                               
  │ 5   │ Joint P(accumulate AND survive 40yr) │ The one number that matters                                │                                                               
  ├─────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ 6   │ Glide path in decumulation           │ 40yr needs different stock/bond mix than 20yr              │                                                               
  ├─────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ 7   │ Historical stress scenarios          │ What 2000 or 2008 would have done to your retirement at 54 │                                                               
  ├─────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ 8   │ Student-t returns                    │ Fatter tails = more realistic crash frequency              │                                                               
  └─────┴──────────────────────────────────────┴────────────────────────────────────────────────────────────┘

# ── NEXT SESSION ──────────────────────────────────────────────────────────────

## ✓ P1 · Cash allocation optimiser — monthly split  [DONE]
Golden-section search over stocks/bonds split ratio f maximises income success rate per upfront allocation.
Table shows TRUE best achievable success rate + optimal stocks/mo + bonds/mo per row.

## ✓ P2 · AR(1) inflation model  [DONE]
mc_inflation_ar1_phi=0.6 (config) — phi=0 recovers original model exactly (same RNG path).
~4× fatter tails on cumulative inflation over 15yr → slightly higher required portfolio.


# ── CORE COMPLETENESS ─────────────────────────────────────────────────────────

## ✓ P3 · Decumulation MC — 35–40 year horizon  [DONE]
User retires at 54. No pension. Portfolio must last 35–40 years, not 20–25.
What to build:
  - simulate_decumulation(portfolio, monthly_withdrawal, n_years, stock_std, ...) → MC over withdrawal phase
  - Fixed withdrawal baseline: 3.3% SWR, inflation-adjusted (already in config)
  - Variable withdrawal / Guyton-Klinger guardrails:
      cut withdrawals if portfolio falls >20% below inflation-adjusted baseline
      raise withdrawals if portfolio rises >20% above → dramatically improves 40yr survival rate
  - Chain to accumulation: p50 and tail_pct (p3) portfolios from accumulation MC feed as
    starting values for decumulation → joint P(accumulate enough AND survive 40 years)
  - Decumulation horizon config field: mc_decumulation_years (default 40)
Note: update TODO note — horizon is 35–40yr (age 54→94), not 30yr as previously written.

## ✓ P4 · SORR visibility in decumulation  [DONE — see section 4.5]
Sequence-of-returns risk is the dominant risk at age 54: a crash in year 1–5 of retirement
permanently impairs the portfolio even if long-run returns are fine.
Show in decumulation report: success rate conditioned on a bad first decade (e.g. -30% in yr 1).
Historical scenario: overlay 2000–2002 and 2008–2009 sequences on the decumulation MC.

## ✓ P5 · Joint probability — P(accumulate AND survive 40yr)  [DONE — see section 4.6]
Single most important output for retirement planning. No classical tool computes this.
Once P3 is done: chain accumulation MC output → decumulation MC input, report joint probability.
The number to headline: "P(retire by 2041 AND portfolio lasts to age 94) = X%"


# ── MAKES IT MORE ADVANCED THAN CLASSICAL TOOLS ───────────────────────────────

## P6 · ETF → CAGR bridge (link main_cagr_analysis.py to main simulation)
main_cagr_analysis.py already fetches real historical data. Missing: the link.
Config field: etf_ticker = "VWCE.DE" → auto-set expected_yearly_growth_stocks_percent
and mc_stock_return_std_percent from historical rolling mean CAGR and ann_vol_eur.
Makes the core assumption evidence-based instead of a manual guess.
Must come before decumulation so inputs are credible.

## ✓ P7 · Student-t return distribution (fat tails)  [DONE]
Config field: mc_stock_return_df (default 1000 = normal, 5 = fat tails).
Cholesky + chi² transform preserves target std; crashes are more frequent and deeper.
Active in config.toml: mc_stock_return_df = 5.0

## ✓ P8 · Stock/bond return correlation  [DONE]
Config field: mc_stock_bond_correlation (default 0.0). Cholesky on joint z-scores.
Active in config.toml: mc_stock_bond_correlation = -0.2 (typical flight-to-quality hedge).

## ✓ P8b · Stock return mean reversion (AR(1))  [DONE]
Config field: mc_stock_return_ar1_phi (default 0.0, active -0.25).
AR(1) in z-space: marginal distribution remains exactly t(df); post-crash recovery is modelled.
Active in config.toml: mc_stock_return_ar1_phi = -0.25

## ✓ P8c · Parameter uncertainty  [DONE]
Config fields: mc_stock_mean_uncertainty (default 0.0, active 0.015 = ±1.5pp on CAGR),
mc_stock_vol_uncertainty (default 0.0, active 0.03 = ±3pp on vol).
Each simulation path draws its own μ and σ from a prior — models epistemic uncertainty
about the true long-run CAGR and volatility. Widens outcome distribution honestly.

## P9 · Historical stress scenarios (accumulation)
Run accumulation with real return sequences: 2000–2002 crash, 2008–2009, 2022 drawdown.
Data already available via main_cagr_analysis.py / yfinance.
Shows: "if you had started investing in the worst year of the last 30, would your plan survive?"
Separate from MC — deterministic but uses real data. Very concrete and relatable.

## ✓ P10+P11 · Glide path + allocation optimiser  [DONE — see section 4.7]
Annual rebalancing in _run_decum_loop: stock% declines linearly from starting allocation
(= 1 - bonds_pct) to mc_decumulation_glide_end_pct over mc_decumulation_glide_years years.
Joint sweep (compute_allocation_sweep) tests bonds_pct ∈ [20%–80%], same random draws,
reports survival at yr20/30/35/40 for fixed and GK, optimal = max yr-30 GK survival.
IT result: bond-heavy allocations win due to 12.5% bond tax (vs 26% stocks).


# ── NICE TO HAVE ──────────────────────────────────────────────────────────────

## P13 · HTML/PDF report export
Makes the report shareable. Generate from the existing terminal output + summary.

