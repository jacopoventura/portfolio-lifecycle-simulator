# Methodology

Reference document explaining what each report section computes, what assumptions and shortcuts the Monte Carlo (MC) engine makes, and — critically — **what the report does NOT model**. Read this before interpreting any number in production.

The project is run **regularly** (monthly–quarterly). Numbers shift; this document explains which shifts mean something.

---

## 1. The canonical FI required portfolio

**The single number that drives the entire report.**

`W*` is the smallest starting wealth at retirement such that, when withdrawing the inflation-adjusted target income (`target_monthly_retirement_income × (1 + expected_inflation_percent)^years` × 12 → annual nominal), **at the optimal bonds allocation**, both these hold:

| Condition | Default config |
|---|---|
| **Baseline preservation**: `P(final ≥ X × W* \| random returns) ≥ Y` | `fi_preservation_ratio = 0.50`, `fi_preservation_confidence_baseline = 0.95` |
| **Stress preservation**: `P(final ≥ X × W* \| worst-SORR returns) ≥ Z` | `fi_preservation_confidence_stress = 0.50` |

The worst-SORR scenario is picked dynamically — at the upper search bound (€10M), the SORR sequence with the lowest `P(preserve ≥ 50%)` becomes the binding stress. With current defaults this is **"Forced crash (yr 1-5)"** in most configurations.

### What this means in plain English

> *"Starting with wealth W*, the optimal allocation gives me at least a 95% chance of preserving half my wealth under random markets — AND at least a 50% chance of preserving half my wealth even if the first 5 years are a market crash."*

If you can't clear both bars, the FI plan is not preservation-calibrated. Survival (balance > €0) is a permissive bar that doesn't justify equity risk on its own.

### Why this is the right bar

- Survival ≠ FI: a 95% survival plan that ends year 30 with €1 left is functionally identical to a cash-spending plan with extra volatility. Investing in equities is only justified if it **preserves real wealth**.
- Two bounds matter: baseline (median market behavior) and stress (the bad scenario you can't talk yourself out of caring about).
- Allocation is **optimized per row** so the bar is what's achievable under the best allocation, not what current allocation happens to deliver.

### Where it's computed

`compute_canonical_fi_required_scan()` in `monte_carlo.py`. Outputs:
- A row per X ∈ {50%, 60%, …, 100%}: scan of `W*(X)` and required bonds %
- A "canonical" row at `X = fi_preservation_ratio` — this is the FI bar used everywhere.

Called once in `main.py:prep_canonical_fi()` before either pillar runs; result threaded into accumulation via `--target-required-portfolio` CLI and into decumulation via `canonical_fi_scan=` kwarg.

---

## 2. Section-by-section reference

### Pillar 1 — Accumulation (`main_portfolio_projection.py`)

**Answers**: "Can I reach the FI canonical bar by my retirement date?"

**Inputs**: current portfolio, monthly net saving, target retirement income, planning CAGR, horizon `years`.

**Outputs**:
- Required portfolio at horizon (sized to the FI canonical bar, NOT the income formula)
- MC success rate (P(both stocks and bonds meet split required at horizon))
- MC income success rate (P(portfolio income ≥ target))
- Plan-B contributions for various CAGR scenarios
- Optimal cash redeployment (where to put the upfront €X)
- Sensitivity to inflation, CAGR, contribution rates
- Inflation × CAGR heatmap

**Two scenarios shown**: DE (German tax regime, Teilfreistellung 0.7) and IT (Italian, no Teilfreistellung). The user is IT-resident; DE is a comparison reference.

**Tax model in accumulation**:
- Stocks: capital-gains tax at sale only (no realised gains during accumulation in the MC; deterministic projection in `finance.py:run_projection` does track Vorabpauschale for DE)
- Bonds: coupon-style taxation at `tax_rate_bond_italy = 12.5%` for IT, `tax_rate = 26%` for DE
- Stock gain fraction (used for the IT income-formula sizing): `(final_stock_value - cost_basis) / final_stock_value` at horizon — the **at-retirement** value (not the 0.65 horizon-averaged proxy that decum uses)

### Pillar 2 — CAGR realism + ETF picks (`main_cagr_analysis.py`)

**Answers**: "Is my assumed CAGR realistic? Which ETFs are likely to deliver it?"

**Inputs**: historical returns (yfinance), config CAPE, optimizer strategy.

**Outputs**:
- Rolling-window CAGR distribution (multiple horizons)
- Drawdown analysis
- CAPE-implied forward CAGR (with mean reversion)
- ETF portfolio optimizer (Sortino / Sharpe / min variance / max CAGR / max P50 CAGR / max P1 CAGR / equal weight)
- Plausibility check: does the optimizer's blend match the config CAGR?

**Not validated by historical replay**: most UCITS ETFs in `optimize_tickers` were launched 2008–2016, so backtesting the recommendation through the 2000–2002 lost decade or 2008 GFC is impossible without proxy indices (which introduce TER, tracking error, FX, dividend-timing caveats). See TODO 2.1 for the dropped backtest plan.

### Pillar 3 — Decumulation (`main_decumulation.py`)

**Answers**: "Does the target portfolio survive AND preserve wealth across retirement, even under stress?"

The **report is preservation-led**; survival rates are footnoted, not headlined.

**Subsections**:

| # | Title | Question answered |
|---|---|---|
| 4.1 | Required Portfolio at Retirement | What's the FI canonical bar? How does the planned wealth compare? |
| 4.2 | Survival Curves | P(balance > €0) by year, fixed vs GK |
| 4.3 | Portfolio Fan | P3/P25/P50/P75/P97 wealth percentiles by year (fixed strategy) |
| 4.4 | Monthly Withdrawal Range (GK) | P10/P50/P90 nominal monthly spending under guardrails |
| 4.5 | SORR Stress Scenarios | Survival hit under three historical bad sequences |
| 4.7 | Optimal Stock/Bond Allocation (sweep) | Best allocation across 20%–80% bonds, at the canonical bar |
| 4.8 | Chained Accumulation → Decumulation | End-to-end joint probability with two thresholds (decum-min + accum income-formula) |
| 4.9 | Withdrawal Rate Sensitivity | Survival vs SWR (2.0–4.0%) at current wealth |
| 4.10 | Inflation × CAGR Sensitivity | 2D survival grid |
| 4.11 | **Wealth Preservation under SORR** | **The headline**: P(preserve ≥ X% of starting wealth) for X ∈ {50%, …, 110%} under each SORR scenario, best allocation per cell |
| Takeaway | Verdict / Top Risks / Levers / Status | Preservation-led one-pager |

§4.6 (independent joint probability) was removed — superseded by §4.8 (chained MC) which doesn't assume independence between accumulation and decumulation phases.

---

## 3. Monte Carlo engine — what's modelled

The MC return generator (`monte_carlo._generate_mc_returns`) draws **annual** stock returns, inflation, and bond yields with the following structure:

### Return generation methods

The engine supports two stock return generation methods, selected by `mc_return_method`:

**Parametric** (`"parametric"`, default): Student-t distributed with `df = mc_stock_return_df` (config: 5.0 → fat tails). Mean = `planning_cagr` (or override). Std = `mc_stock_return_std_percent / 100`. Includes AR(1) mean reversion, Cholesky stock-bond correlation, and epistemic parameter uncertainty. Best for: exploring "worse than history" tail scenarios, sensitivity analysis with clean parameter control.

**Block bootstrap** (`"bootstrap"`): Circular block bootstrap from 99 years of spliced S&P 500 annual returns (SPY total return 1993–2026 + ^GSPC price-only 1928–1992 with a constant dividend adjustment from the overlap period). Draws contiguous blocks of `mc_bootstrap_block_size` years (default 5, capturing business-cycle persistence), wrapping at the array boundary. When `mc_bootstrap_mean_shift = true` (default), all bootstrapped returns are shifted by a constant so the sample mean matches `planning_cagr`, preserving variance/skewness/kurtosis/autocorrelation while targeting the user's assumed return level. Best for: realistic crash clustering, volatility regimes, and fat tails derived from actual market history rather than parametric assumptions.

Both methods produce identical output: `(stock_returns, inflations, bond_yields)` as three `(n_simulations, years)` arrays. All downstream code (decumulation loops, SORR patching, grids) works identically regardless of method.

**Bonds and inflation are always parametric** in both modes — no historical CPI or bond yield series are bootstrapped.

### Returns (parametric mode)

- **Stocks**: Student-t distributed with `df = mc_stock_return_df` (config: 5.0 → fat tails). Mean = `planning_cagr` (or override). Std = `mc_stock_return_std_percent / 100`.
- **Bonds**: Normal. Mean = `yield_bonds / 100`. Std = `mc_bond_yield_std_percent / 100`.
- **Inflation**: Normal with AR(1) autocorrelation. Mean = `expected_inflation_percent / 100`. Std = `mc_inflation_std_percent / 100`. Persistence `phi = mc_inflation_ar1_phi` (config: 0.6, calibrated on historical CPI persistence).

### Correlations

- **Stock–bond correlation**: `mc_stock_bond_correlation` (config: −0.2, typical hedge behavior). Applied via Cholesky decomposition during draws.
- **Stock return AR(1) phi**: `mc_stock_return_ar1_phi` (config: −0.25, empirical post-crash mean reversion).

### Parameter uncertainty (epistemic)

- **Stock mean uncertainty**: `mc_stock_mean_uncertainty` (config: 0.015 = ±1.5pp). Each MC path samples a CAGR from a normal around `planning_cagr` with this std, simulating that we don't know the true expected return.
- **Stock vol uncertainty**: `mc_stock_vol_uncertainty` (config: 0.03 = ±3pp). Similarly for volatility.

These two model the fact that historical estimates of mean and vol are themselves uncertain.

### Tax model in decumulation

- **Bonds**: Italian regime — net yield = `yield_bonds × (1 - tax_rate_bond_italy)` = `0.031 × 0.875 = 2.71%`. Applied to bond principal each year.
- **Stock withdrawals**: gross-up to cover capital-gains tax. Effective stock tax rate = `0.65 × decum_teilfreistellung × tax_rate`.
  - `0.65` is a **horizon-averaged gain-fraction proxy** — over a 30-year decumulation, roughly 65% of each stock sale is taxable gain on average (cost basis erodes faster than value as withdrawals progress; first-year sale has lower gain fraction, last-year sale much higher; 0.65 is a defensible midpoint).
  - `decum_teilfreistellung = 1.0` for IT (no Teilfreistellung in Italy) — meaning effective stock tax = `0.65 × 1.0 × 0.26 = 16.9%` per withdrawal.
- **Pension income**: tax-free in the model (already nominal-net per `mc_decumulation_pension_monthly`). Reality: Italian pensions are taxed; the assumption is the user nets the configured value.

### Withdrawal strategies

- **Fixed**: each year withdraw `target_monthly_nom × cumulative_inflation × 12` (real income preserved). Pension reduces stock/bond draw after `mc_decumulation_pension_start_year`.
- **Guyton-Klinger guardrails**: above + multipliers. If portfolio falls below 80% of the inflation-adjusted baseline, multiply target by 0.90. If portfolio above 120%, multiply by 1.10. Multipliers cumulative, clipped to [0.85, 1.20]. Adaptive — protects against bad sequences at the cost of spending volatility.

### Glide path

- `mc_decumulation_glide_years > 0` triggers a linear shift from the initial stock allocation to `mc_decumulation_glide_end_pct` over that many years.
- Rebalancing happens annually after stock growth, before withdrawals.

### Pension

- Monthly amount `mc_decumulation_pension_monthly`, starting at year `mc_decumulation_pension_start_year`.
- Annual growth `mc_decumulation_pension_adjustment_rate` (config: 1.5% nominal — under typical CPI of 2.3%, this is real-negative; the pension's real value erodes slowly).
- Treated as **deterministic** (no reform risk modelled — see "What's NOT modelled" below).

### SORR sequences

Three patched stress scenarios applied to the first N years of stock returns:

| Label | Years | Pattern |
|---|---|---|
| Forced crash (yr 1-5) | 5 | −30%, −15%, +10%, +5%, +8% |
| Historical 2000-2002 | 3 | −9%, −12%, −22% (dot-com bust) |
| Historical 2008-2009 | 2 | −37%, +26% (GFC) |

The **"worst"** at any given starting wealth is picked dynamically: whichever sequence gives the lowest preservation P at that wealth and current allocation.

---

## 4. Key definitions

| Term | Definition |
|---|---|
| **Survival** | Portfolio balance > €0 at year `mc_decumulation_years`. Permissive bar — €1 left counts. |
| **Preservation** | Final wealth ≥ X% × initial wealth (X configurable, default 50%). Stricter — quantifies bequest / margin / safety. |
| **Canonical FI bar** | The smallest wealth that achieves preservation thresholds under both baseline AND worst-SORR. The number that matters. |
| **MC fixed minimum** | Smallest wealth that achieves `mc_target_success_percent` survival under fixed withdrawals. **Diagnostic only** — survival is permissive. |
| **MC GK minimum** | Same with Guyton-Klinger guardrails. |
| **Accum income-formula required** | Wealth such that `bonds·yield_net + stocks·SWR_net = target_annual_net_income`. Used for accum's §1 sizing **before** the canonical override took over; kept as a fallback / secondary anchor in §4.8. |
| **Joint survival (chained MC)** | P(accumulate AND survive decumulation) using one continuous Monte Carlo path. No independence assumption. The honest joint number. |
| **Gain fraction (at-retirement)** | `(final_stock_value - cost_basis) / final_stock_value` at horizon. Used by accumulation. |
| **Gain fraction (horizon-averaged proxy)** | `0.65` — a midpoint estimate of average gain fraction over a 30-year decumulation. Used by decum MC tax. |
| **Worst-SORR** | The SORR sequence with lowest preservation P at a given starting wealth + allocation. Dynamic — picked per run. |
| **FI canonical bar reachable** | Whether `W*` falls within the binary-search range [€100k, €10M]. If `False`, the canonical bar is too high to be computed and the report flags it. |

---

## 5. Non-obvious numerical choices (and why)

### Why `0.65` for the gain-fraction proxy in decum?

A 30-year withdrawal sequence on a growing-then-declining stock balance has an average ratio of (price − cost basis) to price roughly in the 0.55–0.70 range, depending on growth assumptions, withdrawal timing, and reinvestment. `0.65` is a defensible midpoint. The accumulation side uses a different value (the **at-retirement** value, ≈ 0.40 for a 15y accumulation) — there's a documented ~3-4% discrepancy between the two pillars when both use the income formula, but the canonical FI bar resolves it by making preservation the bar, not income coverage.

### Why `0.70` factor in `calculate_yearly_vorabpauschale`?

That's the German **Vorabpauschalfaktor** fixed by §18 InvStG (Investment Tax Act): Basisertrag = Marktwert × Basiszins × 70%. It is **a legal constant**, not `config.teilfreistellung`. Both happen to be 70% for equity ETFs, but they're separate concepts. Teilfreistellung is applied separately at the call site (`finance.py:352`).

### Why `decum_teilfreistellung = 1.0` (default)?

Italy has no Teilfreistellung. The German equity-ETF discount doesn't apply. Setting this to 1.0 means full taxation of stock capital gains during decumulation. The user can override to 0.7 if decumulating under German rules.

### Why AR(1) `phi = 0.6` for inflation?

Italian / Eurozone CPI shows positive serial correlation (high-inflation regimes persist). A 1-year AR coefficient around 0.6 matches empirical fitting on post-1990 data.

### Why `mc_stock_return_df = 5.0`?

Student-t with 5 degrees of freedom gives fat tails (kurtosis ≈ 9 vs Normal's 3). Captures crash days like 1987, 2020. `df → ∞` would be Normal; `df = 5` is a reasonable mid-tail-thickness for monthly-aggregated annual returns.

### Why GK multipliers `[0.85, 1.20]`?

Standard Guyton-Klinger guardrails (2006 paper). 80% lower bound on portfolio → cut 10%; 120% upper → raise 10%; clipped to [85%, 120%] of the inflation-adjusted baseline. Tight enough to track inflation; loose enough to absorb a moderate drawdown.

### Why `fi_preservation_confidence_stress = 0.50` (loose default)?

Stress floor is "50% probability of preserving 50% of initial wealth under the worst SORR sequence". This is a deliberately permissive floor — the worst SORR is, by design, rare. Tightening to 0.95 would make almost no plan viable. The default treats stress survival as a "tail not made worse by panic" check, not a "guaranteed preservation" requirement.

### Why search bounds [€100k, €10M] for the canonical scan?

Spans the plausible range for an Italian early-FI portfolio. Below €100k, no reasonable target income is sustainable; above €10M, the user has other questions. Tolerance 5,000 € closes the bisection in ~14 iterations.

---

## 6. What is NOT modelled

Crucial for honest interpretation. The report's numbers assume **none of the following risks occur unless explicitly listed in stress sweeps**:

### Tail risks not in the SORR sequences

- **Hyperinflation**: only the highest CPI row in §4.10 (4%) is meaningful; sustained 6%+ inflation isn't modelled.
- **Negative real returns over decades** (1970s style): only partially captured by 2000-2002 patch.
- **Multi-regime persistence**: a 10-year bear market isn't an SORR scenario.
- **Black swans not in history**: regulatory chaos, currency collapse, war effects.
- **Property crashes**: real estate doesn't appear anywhere.

### Tax / regulatory drift

- **IT pension reform risk**: pension is deterministic. Future cuts (multiplicative haircuts, age increases) not modelled. See TODO 4.3.
- **Capital-gains tax changes**: `tax_rate = 0.26` is constant. Italian tax history shows shifts every 5-10y. Not stressed.
- **IT bond tax changes**: `tax_rate_bond_italy = 0.125` is constant.
- **Teilfreistellung erosion**: hypothetical EU directive narrowing the discount not modelled.
- **Tax-treaty changes** (IT-DE, IT-LU for fund domicile): not modelled.

### Cost drag not in the model

- **ETF tracking error**: optimizer recommends indices conceptually; real ETFs lag.
- **TER**: implicitly absorbed into `expected_yearly_growth_stocks_percent` per user-supplied assumption — verify it's TER-net.
- **Bid-ask spread**: not modelled.
- **Dividend-reinvestment timing**: instantaneous in the model; reality has 1-2 day lag.
- **Currency conversion costs** for EUR/USD ETFs: not modelled.
- **Tax-loss harvesting / loss carryforward**: simplifications in the favor of conservatism.

### Income / spending dynamics

- **Lifestyle inflation different from CPI**: target income inflates at CPI; reality is often higher (healthcare especially).
- **Healthcare cost escalation in retirement**: not modelled. Italian SSN is mostly free but supplementary coverage isn't.
- **Property purchases / sales / downsizing**: not modelled.
- **Other income** (consulting, part-time, rental): not modelled — would increase preservation.
- **Inheritance** (incoming or outgoing): not modelled.
- **Salary correlation with markets**: contributions are deterministic; in real crashes, layoffs happen. See TODO 4.2 `salary_beta`.

### Longevity

- **Horizon truncation**: `mc_decumulation_years = 30` means decum ends at age 84 (current config). Italian life expectancy at 54 is ~83 for men, ~88 for women — a meaningful fraction of healthy 54-year-olds outlive a 30-year plan.
- **Survivor's pension** / spousal mechanics: not modelled.
- **Annuity options**: not modelled.

### Behavioral

- **Withdrawal-rule discipline**: GK assumes the user actually follows the cut-spend rule. Real people often don't.
- **Allocation rebalancing discipline**: model assumes annual rebalancing; reality has friction.
- **Panic selling in drawdowns**: not modelled.

### When in doubt

If a number sounds reassuring (high preservation P, comfortable surplus), look at the **what's not modelled** list above before declaring victory. Many of these risks are real but uncomfortable to quantify. The right response to a comfortable verdict is to build buffer beyond the FI bar, not to reduce target spending.

---

## 7. Config parameter reference

### Required (no defaults — must be in `config.toml`)

| Param | Used in | Meaning |
|---|---|---|
| `target_monthly_retirement_income` | All pillars | Today's € target monthly net income in retirement |
| `years` | All pillars | Accumulation horizon (years from now to retirement) |
| `expected_inflation_percent` | All pillars | Mean CPI for both projection and MC inflation draws |
| `expected_yearly_growth_stocks_percent` | All pillars | Net-of-TER stock CAGR baseline (overridden by `--cagr` CLI) |
| `bonds_percentage` | All pillars | Allocation split at retirement (0–100) |
| `yield_bonds` | All pillars | Annual bond yield (net of fund expense) |
| `tax_rate` | All pillars | Capital-gains rate (Germany / Italy stocks) |
| `withdrawal_percentage_stock` | §1, §4.9, §4.10 | Stock SWR (used for income formula + sweep grid) |
| `current_stock_value`, `current_bonds_value`, `current_cash` | §1 | Today's portfolio |
| `annual_net_saving` | §1 | Net monthly savings × 12 |

### Decumulation-specific

| Param | Default | Meaning |
|---|---|---|
| `mc_decumulation_years` | 30 | Decum horizon. Retirement age 54 → ends at 84. |
| `mc_decumulation_pension_start_year` | 15 | First year pension kicks in (yr-15 → age 69). |
| `mc_decumulation_pension_monthly` | 2,500.0 | Nominal net monthly pension at start. |
| `mc_decumulation_pension_adjustment_rate` | 0.015 | Annual pension growth (1.5% nominal — slightly below CPI). |
| `mc_decumulation_stock_pct` | 1 − bonds_percentage/100 | Stock fraction at decum start. |
| `mc_decumulation_glide_end_pct` | 0.30 | Stock fraction at end of glide path. |
| `mc_decumulation_glide_years` | 20 | Linear glide duration. |
| `decum_teilfreistellung` | 1.0 (IT) | Teilfreistellung for decum. Set 0.7 if German-resident in retirement. |
| `mc_decumulation_n_simulations` | 10,000 | MC paths for decumulation. ±0.5pp accuracy. |

### FI canonical bar

| Param | Default | Meaning |
|---|---|---|
| `fi_preservation_ratio` | 0.50 | X — preservation threshold (final wealth ≥ X × initial). |
| `fi_preservation_confidence_baseline` | 0.95 | Y — required confidence under random returns. |
| `fi_preservation_confidence_stress` | 0.50 | Z — required confidence under worst SORR. Loose by design. |

### MC return model

| Param | Default | Effect |
|---|---|---|
| `mc_stock_return_df` | 5.0 | Student-t degrees of freedom. Larger = thinner tails. |
| `mc_stock_bond_correlation` | −0.2 | Cholesky correlation. 0 = independent. |
| `mc_stock_return_ar1_phi` | −0.25 | AR(1) on stock returns. Negative = post-crash mean reversion. |
| `mc_stock_mean_uncertainty` | 0.015 | Std of normal prior on CAGR (epistemic uncertainty). |
| `mc_stock_vol_uncertainty` | 0.03 | Std of normal prior on vol (epistemic uncertainty). |
| `mc_inflation_ar1_phi` | 0.6 | Inflation autocorrelation. 0 = i.i.d. CPI. |

### Block bootstrap

| Param | Default | Effect |
|---|---|---|
| `mc_return_method` | `"parametric"` | `"parametric"` or `"bootstrap"`. Selects stock return generation method. |
| `mc_bootstrap_block_size` | 5 | Contiguous years per block. 5 captures business-cycle persistence. |
| `mc_bootstrap_mean_shift` | `true` | Shift bootstrapped mean to match `planning_cagr`. Preserves distribution shape. |

### CAGR / ETF (`[etf_optimizer]` block)

See `config.toml` comments — strategy choice, ticker universe, weight caps. Not detailed here because the §2 module is mature and self-explanatory.

---

## 8. When numbers shift between runs

The drift DB (TODO 2.2) will eventually surface this automatically. Until then, common reasons numbers shift:

| Shift | Likely cause |
|---|---|
| Canonical FI bar jumps | `expected_inflation_percent` or `target_monthly_retirement_income` changed |
| Preservation P drops 5pp+ | New MC seed; sampling noise (~±2pp on 10k sims); check git log for engine changes |
| Worst-SORR label changes | Allocation shifted; different scenarios now bind |
| §4.5 survival rate jumps | Tax rate, pension parameter, or glide-path config changed |
| §1 IT scenario diverges from previous run | `tax_rate_bond_italy`, `teilfreistellung`, or `withdrawal_percentage_stock` changed |
| Optimizer picks different ETFs | CAPE updated, ticker history grew, weight caps changed |

Run `git diff config.toml` first when investigating drift.

---

## 9. Quick-reference: the verdict logic

The takeaway block decides "am I on track?" with this logic:

```
gap_to_fi = planned_wealth - canonical_FI_bar
worst_stress_p = min(preservation P at worst-SORR scenario, 50% threshold)

if gap_to_fi >= 0 and worst_stress_p >= fi_preservation_confidence_stress:
    "On track: planned wealth clears the FI canonical bar and worst-SORR floor."
elif gap_to_fi >= 0 and worst_stress_p < fi_preservation_confidence_stress:
    "On track for baseline, but FRAGILE under worst-SORR."
else:
    f"Off track: need €{-gap_to_fi} more wealth."
```

Survival rates appear only as a one-line footnote. Survival is a permissive bar; FI is not.
