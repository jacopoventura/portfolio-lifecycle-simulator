# Project TODO

## Goal

Answer **"Am I on track to financial freedom? If not, why, and what should I adjust?"**

Project pillars:

1. **Accumulation** (`main_portfolio_projection.py`) — required portfolio at horizon, contributions, success probability
2. **CAGR realism + ETF picks** (`main_cagr_analysis.py`) — does the assumed return match history? Which ETFs realise it?
3. **Decumulation feasibility** (`main_decumulation.py`) — does the target portfolio survive AND preserve wealth across the horizon?
4. *(Planned)* **FI date** (`main_fi_date.py`) — **when** can I retire? Inverse view + goal funnel.

Project is run **regularly** (monthly–quarterly) with updated config, so reproducibility and drift detection matter.

---

## Current state

| Pillar | Status | Latest changes |
|---|---|---|
| §1 Accumulation | Mature | Targets the canonical FI bar (preservation-calibrated). Both DE and IT scenarios accept `--target-required-portfolio X` override. Plan-B math, success rates, and contribution sizing all use the canonical bar. |
| §2 CAGR + ETF | Mature | Historical CAGR windows, drawdown, CAPE-implied returns, ETF optimiser (multiple strategies), plausibility check. **Not yet backtested empirically — Priority 2.1.** |
| §3 Decumulation | Mature + preservation-led | Survival curves, SORR, allocation sweep, chained MC, SWR sensitivity, **§4.11 Wealth Preservation under SORR**, takeaway block. Verdict is preservation-led (baseline + worst-SORR P(preserve)); survival is a footnote. |
| §4 FI date | Not built | Priority 4.1 — answers "when can I retire?" |

Latest architectural change: pipeline now computes a **canonical FI bar** (preservation-calibrated required wealth) BEFORE either pillar runs, then passes it as the target for both accumulation (`--target-required-portfolio X`) and decumulation (anchor + display). Single number, single verdict.

---

## ✅ Priority 1 — Coherence — DONE

### 1.1 Cross-pillar required-portfolio reconciliation ✅
Implemented as a 4-phase rollout:

- **Phase 1**: defined `compute_canonical_fi_required_scan()` in `monte_carlo.py`. New config params `fi_preservation_ratio` (default 0.50), `fi_preservation_confidence_baseline` (0.95), `fi_preservation_confidence_stress` (0.50). Scans wealth-vs-confidence curve for X ∈ {50%, 60%, …, 100%}; canonical = the row matching `fi_preservation_ratio`.
- **Phase 2**: decumulation `required_anchor` now = canonical FI wealth. §4.1 rewritten to lead with the canonical bar; old MC fixed minima are diagnostic-only.
- **Phase 3**: `main.py` runs `prep_canonical_fi()` before either pillar, then threads the canonical wealth into accum (`run_accum(target_required_portfolio=…)`) and decum (`run_decum(canonical_scan=…)`). `main_portfolio_projection.py` gained `--target-required-portfolio` CLI override; `finance.py:run_scenario` accepts `required_total_portfolio_override` that recomputes dependent fields (stock/bonds amounts, monthly contributions, generated income).
- **Phase 4**: takeaway VERDICT reframed around preservation. Baseline + worst-SORR P(≥50%) and P(≥100%) lead; survival is one footnote line. Verdict prose: on-track / fragile-under-SORR / off-track.

`DecumulationSummary.required_portfolio_fi_canonical` is the user-facing single number for the FI bar.

### 1.2 Reframe takeaway around preservation ✅
Done as part of Phase 4 above.

---

## Priority 2 — Validation (does the recommendation actually work?)


### 2.2 Drift / run-history DB
You run this regularly. Each run currently leaves no trail beyond `historical_data/last_run_*` (overwritten). For regular use this is the highest-leverage missing feature.

**Action:** append-only SQLite at `historical_data/history.db` storing per-run:
- timestamp, config hash, git commit
- key output numbers: canonical FI bar, accum P50/P5, §4.11 P(preserve 50%) baseline+worst, required_cagr, recommended ETF weights, planning_cagr_override

At report start: query last run with same config hash; if differs, print a delta row:
```
DRIFT vs last run (2026-04-15, commit abc1234):
  FI canonical bar:      €2.14M → €2.18M  (+€40k)
  P(preserve 50%) base:   95.5% → 96.2%   (+0.7pp)
  P(preserve 50%) worst:  83.7% → 79.4%   (−4.3pp)  ← worst-case has gotten worse
  required_cagr:          5.8% →  5.4%    (−0.4pp)
```

Embed `config_hash + git_commit` in every report header so any saved output is traceable to its inputs.

**Effect:** answer "have my numbers shifted?" without manually diffing two reports.

---

## ✅ Priority 3 — Reliability — DONE

### 3.1 METHODOLOGY.md ✅
**Done**: `METHODOLOGY.md` at project root. Nine sections covering:
1. The canonical FI required portfolio (the single number driving the report)
2. Section-by-section reference (what each pillar answers + inputs + outputs)
3. MC engine — what's modelled (returns/inflation/vol/correlation/tax/pension/withdrawal/glide/SORR)
4. Key definitions glossary (survival vs preservation, MC-min vs canonical, gain-fraction)
5. Non-obvious numerical choices and why (0.65 proxy, 0.70 Vorabpauschalfaktor, AR(1) phi, GK guardrails, search bounds)
6. **What is NOT modelled** — comprehensive list (tail risks, regulatory drift, cost drag, income/spending dynamics, longevity, behavioral)
7. Config parameter reference
8. When numbers shift between runs (drift causes)
9. Quick-reference for the verdict logic

### 3.2 Property tests ✅
**Done**: `tests/test_properties.py` — 16 invariant tests, hand-rolled (no hypothesis dependency).
Categories: bounds (4), time monotonicity (2), wealth monotonicity (2), withdrawal monotonicity (1), inflation monotonicity (1), pension monotonicity (2), canonical FI monotonicity (2), cross-section consistency (2). Each uses small n_simulations + fixed seed so monotonicity holds deterministically (RNG draws shared across the comparison runs).

### 3.3 Fixture regression tests ✅
**Done**: `tests/test_fixtures.py` — 5 snapshot tests pinning canonical wealth, survival rates, chained joint P, baseline preservation P, bridge milestone.
Snapshots generated via `tests/_fixture_probe.py` (rerun + bake values into `tests/test_fixtures.py` when MC engine changes intentionally). Tolerances tight: ±1pp on probabilities, ±0.5% on wealth amounts.

### 3.4 CI import smoke ✅
**Done**: `tests/test_import_smoke.py` — `pytest.mark.parametrize` over every discoverable `*.py` module in the project (currently 21). Catches SyntaxError/ImportError/NameError at the top level of any module. Auto-discovers — new modules are tested automatically.

---

## Priority 4 — Inverse and tail refinements (advanced)

Priority 4 has **three sub-items with different architectural footprints**:

| Sub-item | Touches | New script? |
|---|---|---|
| 4.1 Inverse problem ("when can I retire?") | New compute path (sweep retirement year), new output style (FI-date distribution + goal funnel) | **Yes — `main_fi_date.py`** |
| 4.2 Block bootstrap MC | The MC engine (`_generate_mc_returns`), config switch, affects every MC | No — modifies existing engine, opt-in via config |
| 4.3 Policy uncertainty (pension cuts, tax shifts) | Decum-only, narrow scope | No — new §4.12 sub-section in `main_decumulation.py` |

Recommended order: **4.3 first** (cheapest, immediate IT-resident value) → **4.1** (highest leverage, answers the actual question) → **4.2** (deepest engine change, refines confidence).

### 4.1 Inverse problem — `main_fi_date.py`

**New top-level script.** Answers three questions in three report blocks:

**A. FI-date distribution** — "When does my MC P50 wealth cross the FI bar?"
```
FI DATE DISTRIBUTION (at current trajectory)
   Confidence    FI Year   Age   Years from now
   P10 (lucky)    2038      51         12
   P50 (median)   2041      54         15  ← config target
   P90 (unlucky)  2046      59         20
```

**B. Year-by-year clearance probability** — per candidate year y ∈ {now, …, now+25}:
```
   Year   Age   FI bar (y-dep)   P50 wealth      P(reach FI)
   2036    49   €2,055k          €1,712k          18.3%
   2041    54   €2,145k          €2,712k          93.6%  ← config target
   2043    56   €2,170k          €3,108k          98.2%
```

**C. Goal funnel** — "to retire 2y earlier (2039) at P90 confidence, you need…"
```
   Lever                  Required adjustment       Realistic?
   Extra contributions     +€820/mo                 ✓ within saving capacity
   Higher CAGR             +1.1pp (6.0% → 7.1%)     ? CAPE check
   Lower target spend      −€280/mo                 ✓ small lifestyle shift
   One-time injection      +€350k upfront           ? (depends on assets)
```

**How it works**:
1. Pre-compute FI bar for each candidate retirement year (canonical depends on decum horizon = lifespan − retirement age).
2. Run one long accumulation MC out to year+25 with per-year wealth distribution.
3. Per MC path, find the earliest year `y` where wealth crosses `fi_bar(y)`. Aggregate P10/P50/P90.
4. Goal funnel uses **deterministic projection** (not full MC) per lever — fast 1D root-finding.

**Files**:
- New: `main_fi_date.py`
- New dataclasses in `models.py`: `FIDateDistribution`, `YearByYearClearance`, `GoalFunnel`
- New: `reports/section5.py` (printer)
- Modified: `monte_carlo.py` — `compute_fi_date_distribution(config, …)`, `compute_goal_funnel(config, target_year, …)`
- Modified: `main.py` — optional `--inverse` flag runs after the standard pillars; `python main_fi_date.py` works standalone

**Effort**: 2–3 days.

### 4.2 Block bootstrap MC (engine swap)

**Where**: `monte_carlo.py` + new `historical_returns.py`.

Replace Gaussian + AR(1) with bootstrap from historical monthly returns. Captures structurally:
- **Fat tails** (1987 −20%, March 2020)
- **Vol clustering** (calm regimes ↔ crisis regimes)
- **Persistence** (2000-2002 lost decade isn't an i.i.d. event)

Then add a regime-switching layer (3-state Markov: bull / bear / lost-decade, calibrated on history).

Finally `salary_beta` parameter — `annual_net_saving` reduced during MC crash years to model human-capital correlation with markets.

**Files**:
- New: `historical_returns.py` — loads monthly returns from a CSV fixture, provides bootstrap samplers (Politis-Romano block bootstrap, block length ~12 months)
- New config: `mc_return_model = "gaussian"` (default) | `"block_bootstrap"` | `"regime_switching"`
- New config: `salary_beta = 0.0` (default) | `0.3-0.6`
- Modified: `monte_carlo.py:_generate_mc_returns` dispatches on `mc_return_model`
- New property tests: bootstrap mean / std reproduce historical sample within tolerance

**Effect**: every MC tail number becomes more realistic. §4.11 worst-SORR P numbers shift (likely down). §1's P5 wealth gets honest about fat tails. The canonical FI bar may shift accordingly.

**Effort**: ~2 days. Affects every MC in the project (good thing — once it works, everything benefits).

**No new script.** Pure engine swap, opt-in via config.

### 4.3 Pension/tax/policy uncertainty (decum extension)

**Where**: new sub-section in `main_decumulation.py` and `reports/section4.py`.

Currently pension is deterministic (€2,500/mo from yr-15) and tax rates are constant. Reality has shifts (Italian pension reforms every ~10 years, capital gains tax revisions).

**§4.12 Policy Uncertainty Sensitivity** — runs the §4.11 preservation matrix under policy stress combinations:

```
4.12  Policy Uncertainty — preservation under regulatory stress

   Policy stress                          P(preserve ≥50%)   P(preserve ≥100%)
   Baseline                                   99.6%               91.7%
   Pension cut −25% (one-off, yr-15)          92.1%               78.4%
   Pension cut −50%                           80.6%               60.2%
   Tax rate +2pp (lifelong)                   95.8%               84.3%
   Pension −25% AND tax +2pp                  85.3%               65.0%
```

New config:
- `policy_stress_pension_cuts = [0.0, 0.25, 0.50]` (multiplicative haircuts applied at pension start year)
- `policy_stress_tax_shifts_pp = [0, 2, 5]` (additive pp shifts to `tax_rate` for the decum simulation)

**Files**:
- Modified: `monte_carlo.py` — new `compute_policy_uncertainty_sensitivity()` that reuses `compute_wealth_preservation_under_sorr` with `dataclasses.replace(config, …)` per stress
- Modified: `reports/section4.py` — new `print_policy_uncertainty()` printer
- Modified: `main_decumulation.py` — wires it as §4.12

**Effect**: explicit visibility into the regulatory tail risks an Italian retiree actually faces — pension reform and tax-rate drift.

**Effort**: ~half day. Reuses existing wealth-preservation infrastructure.

**No new script.**

---

## Smaller items / cleanup

- §4.10 inflation grid is hardcoded `[0.010, 0.015, …, 0.040]`. Make config-driven.
- §1 inflation sensitivity table hardcodes `[1.5, 2.0, 2.3, 2.5, 3.0, 3.5]`. Same.
- Module constants `_SENSITIVITY_N_SIMULATIONS`, `_SUGGESTIONS_N_SIMULATIONS`, `_OPTIMIZE_N_SIMULATIONS`, `_COMBINED_GRID_N_SIMULATIONS`, `_INFLATION_CAGR_N_SIMULATIONS` are tuning knobs — consider exposing as config or documenting why they're fixed.
- `_OPTIMIZE_MIN_CASH_TODAY = 50_000.0` in `monte_carlo.py` is hardcoded; should follow `config.cash_reserve_at_horizon` or become its own config.
- The canonical scan's "worst-SORR" is picked at the upper search bound (€10M); the takeaway's "worst" is picked at the planned wealth. These can legitimately differ. Document or unify.
- `fi_preservation_confidence_stress = 0.50` is over-satisfied at current config (achieved ~86%). Could tighten the threshold, or document why the loose floor is intentional.

---

## Nice to have

- GUI / web frontend
- Email/Slack notification on drift (Priority 2.2) crossing a threshold
- Plotting (PNG output of §4.3 fan chart, §4.11 preservation matrix as heatmap, FI-date P10/P50/P90 funnel)
- Standalone `python main_fi_date.py` runnable independent of `main.py` (once 4.1 lands)
