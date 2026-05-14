remember that the goal of all of this project is to understand if I am on track to financial freedom. If not, I need to understand why and adjust 
(or understand what is missing to reach the goal). It means that not all the numbers from calculations are useful for this goal.
This includes understanding if the target portfolio can survive during the decumulation phase, if the historical market performance match the 
expected cagr used for portfolio calculation and if I can find a ET>F portfolio to reach this


REPORTING / FI-VERDICT FIXES (ranked by FI-impact)
1. Unify the required-portfolio definition. Section 1 uses the asset-split formula (€2.07M IT) and prints "WARNING — financial plan must be
   revised" at 76% success; Plan Verdict uses the income formula (€1.96M) and prints "✓ ON TRACK". Same inputs, contradictory verdicts.
   Pick one canonical target, or print both side-by-side with explicit labels.
2. Fix Plan Snapshot to consume typed values, not regex over rendered text. main.py:65-85 currently extracts "Required portfolio (decum)",
   "Required stock CAGR", "target withdrawal" via regex from the printed report — they all render as "—" because the anchors no longer match.
   run_accum already returns required_cagr/p50 as floats; extend the return tuple to also pass target_withdrawal and required_portfolio_decum
   into print_plan_snapshot directly.
3. Surface Required stock CAGR next to CAPE forward + S&P 10y rolling EUR + optimised ETF P50/P5, all in one row of Plan Snapshot.
   Single-glance realism check: "do I need X% and does X% exist in the real world?".
4. Filter ETF optimiser candidates by P5 != n/a (or flag them). Current best pick (IS3R 19% + QQQ 25% + IGLN 10% + TDIV 46%) shows P5=n/a
   because common history < ~10y. A 40-year decumulation plan should be optimised on P5 survival, not P50.
5. Persist a one-line history log per run (date, required_cagr, joint_P, P50) so drift over time is visible — early warning that the plan
   needs revising.

CODE QUALITY
1. [DONE] Pre-commit hook now byte-compiles every Python file (`compileall`) and runs `tests/test_imports.py` which imports every module
   (catches SyntaxErrors + ImportErrors in lazily-imported files like the one that survived in reports/section4.py).
2. [DONE] Renamed print_summary_q0/q1/q2a/q2b/q4 → print_current_situation_summary / print_what_you_need_at_horizon /
   print_how_to_get_there_portfolio / print_how_to_get_there_income / print_redeploy_current_assets.


ADVANCED / MAKE THE PROJECT OUTSTANDING
1. Risk model honesty — replace Gaussian MC with block bootstrap from historical monthly returns (free fat tails, autocorrelation, vol
   clustering). Then add regime-switching MC (bull/bear/lost-decade Markov, calibrated on history). Finally a `salary_beta` parameter so
   `annual_net_saving` is reduced during MC market crashes — the single most under-modelled FI risk (human-capital correlation).
2. Inverse-problem surface — answer "when can I retire?" instead of "given 2041, on track?".
   - FI-date distribution: P10/P50/P90 of FI year given current trajectory.
   - Goal funnel: "to retire 2 years earlier with same confidence, you need contributions +€X/mo OR CAGR +Ypp OR target income −€Z/mo".
3. Backtesting bridge — take the Section 3 optimiser's recommended ETF portfolio and replay it through actual historical periods
   (1985→2025, 2000→2024 lost-decade entry, 2007→2024 GFC entry). Today the optimiser's pick is never validated against real history.
4. Drift detection + reproducibility — append-only SQLite run-history DB (date, config hash, code commit, required_cagr, joint_P, P50, P5).
   Print a delta report each run vs the previous one. Embed config hash + git commit in every report header so outputs are traceable to inputs.
5. Reliability scaffolding — fixture regression tests (known config → known P50/joint_P ± tolerance, CI fails on drift); property tests
   via hypothesis ("increasing inflation must not increase survival"); import smoke test in CI (`python -c "import <module>"` for each)
   so SyntaxErrors can't survive in lazily-imported files; METHODOLOGY.md explaining what each MC does and does NOT model.


NICE TO HAVE
1. GUI