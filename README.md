# portfolio-lifecycle-simulator

**Goal — model the full investment lifecycle of a portfolio whose job is to
cover living expenses in retirement *beyond* what state or occupational
pensions provide.**

The engine treats the portfolio as a top-up to a pension stream, not as the
sole source of income, and answers five questions end to end:

1. **How much capital is required at the horizon** to fund a target income,
   net of taxes and of expected pension entitlements?
2. **What is the probability of reaching it** with the current portfolio,
   savings rate and assumptions?
3. **What raises that probability most** — contributing more, reallocating,
   redeploying idle cash, or retiring later?
4. **Does the portfolio then survive and preserve its value** across the
   withdrawal phase, including when markets fall in the opening years?
5. **Which instruments plausibly deliver** the return the plan assumes?

Most planning tools stop at *"will the money last?"*. Survival is a permissive
bar: a plan that ends year 30 with €1 left technically succeeded. This engine
is calibrated on **capital preservation under stress** instead — how much real
wealth remains, and how confident one can be about it when the market
misbehaves early.

---

## What it computes

The five questions map onto three pillars:

| Pillar | Question | Core output |
|---|---|---|
| **S1 · Accumulation** | Do contributions reach the required portfolio by the target date? | Success probability, required contributions, optimal cash redeployment, sensitivity grids |
| **S2 · Decumulation** | Does that portfolio survive *and* preserve real value over the withdrawal horizon? | Survival and preservation probabilities, safe withdrawal rate, SORR stress matrices |
| **S3 · Instruments** | Does the real world actually deliver the assumed return? | Rolling-window CAGR distributions, CAPE-implied return, Sortino-optimised ETF mix |

**The canonical requirement `W*`** ties them together: the smallest starting
wealth such that, at the optimal allocation, the portfolio preserves a target
fraction of real capital *both* under ordinary random markets and under a
forced historical crash in the opening years — each at its own configurable
confidence level.

## Modelling

- **Stochastic returns** — 50,000 paths; Student-t innovations for fat tails,
  AR(1) inflation, correlated bond-yield shocks.
- **Sequence-of-returns risk (SORR)** — every result is re-run against forced
  crash sequences (2000–2002, 2008–2009, synthetic year 1–5 crash) so early-loss
  vulnerability is visible rather than averaged away.
- **Withdrawal strategies** — fixed real withdrawals, or Guyton-Klinger
  guardrails (cut 10% below the lower band, raise 10% above the upper).
- **Glide paths** — linear equity drift across the decumulation horizon, in
  either direction.
- **Two-bucket structures** — a never-withdrawn bequest sleeve plus a
  depletion-allowed spending sleeve, compared against a single blended pot at
  the *same* confidence bar.
- **Taxation** — German (Vorabpauschale, Basiszins, Freibetrag,
  Teilfreistellung) and Italian regimes modelled separately, since they change
  the answer materially.

`METHODOLOGY.md` documents every assumption **and what the model deliberately
does not capture**. Read it before trusting a number.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp config.example.toml config.toml     # then edit your assumptions
python main.py --no-gui
```

`config.toml` holds personal figures and is gitignored; `config.example.toml`
is the committed template. Running `python main.py` without `--no-gui` opens a
small dialog to adjust the plan levers first.

Optional: put an `ANTHROPIC_API_KEY` in `.env` to enable the LLM commentary
section (`--agent`); everything else runs without it.

```bash
pytest                                  # unit, fixture and property tests
```

---

## Command line

| Flag | Effect |
|---|---|
| `--no-gui` | Skip the parameter dialog, run `config.toml` as-is |
| `--cagr 4` | Override the planning CAGR for all sections (`4`, `4.0`, `0.04`, `4%`) |
| `--from decum` \| `cagr` \| `agent` | Start from a later section |
| `--verbose` | Include the extended stress and comparison tables |
| `--agent` | Append the LLM commentary section |
| `--<plan-parameter> <value>` | Override any plan lever (see below) |

Passing any plan-parameter flag implies `--no-gui`.

---

## Agent / tool interface

The engine is designed to be driven by an LLM agent (via MCP tools) as well as
by a human. Two properties make that safe:

**Overrides never mutate the real config.** When plan-parameter flags are
passed, the merged configuration is written to a temporary file and the
pipeline is pointed at it through the `FIRE_CONFIG` environment variable. An
agent can explore twenty scenarios and leave the user's actual plan untouched.
Every module resolves its config through `models.resolve_config_path()`.

**Validation is shared and fails loudly.** CLI overrides run through the same
validator as the interactive dialog; on invalid input nothing is written and
the run stops with a message naming the offending parameter.

### Parameter reference

All parameters are optional; anything unspecified keeps its `config.toml`
value. Flag form is the key with underscores replaced by hyphens
(`--target-monthly-retirement-income 3000`).

| Parameter | Unit / range | Meaning |
|---|---|---|
| `target_monthly_retirement_income` | € per month, **today's money** | Net income the portfolio must fund; inflated to the retirement date internally |
| `years` | years, > 0 | Time from now until retirement (accumulation horizon) |
| `expected_yearly_growth_stocks_percent` | % per year, e.g. `6` | Planning CAGR for equities, net of fund costs. Nominal, not real |
| `expected_inflation_percent` | % per year, e.g. `2.3` | Expected inflation; drives all real-terms conversions |
| `bonds_percentage` | % of portfolio, `0`–`100` (**not** 0–1) | Bond share at retirement |
| `yield_bonds` | % per year | Nominal bond yield, net of fund costs |
| `withdrawal_percentage_stock` | % per year | Assumed sustainable withdrawal rate used to *size* the income requirement. Sizing input, not a simulated behaviour |
| `current_stock_value` | € | Equities held today |
| `current_bonds_value` | € | Bonds held today |
| `current_cash` | € | Cash held today |
| `annual_net_saving` | € per year | Total net amount saved annually |
| `stock_monthly_investment_pct` | fraction, `0`–`1` | Share of saving directed to equities |
| `bonds_monthly_investment_pct` | fraction, `0`–`1` | Share of saving directed to bonds. Stock + bond shares must not exceed 1 |
| `mc_decumulation_years` | years | Length of the withdrawal phase |
| `fi_preservation_ratio` | fraction, `0`–`1` | Share of initial real capital that must remain at the end of the horizon. `0.6` = keep 60% |
| `fi_preservation_confidence_baseline` | fraction, `0`–`1` | Required probability of meeting the preservation bar under ordinary random markets |
| `fi_preservation_confidence_stress` | fraction, `0`–`1` | Required probability of meeting it under the worst forced-crash sequence |

### Notes for an agent choosing values

- **Percent vs fraction is inconsistent by design of the underlying config**:
  allocations and rates are percentages (`bonds_percentage = 50`), while
  confidences and saving shares are fractions (`0.9`). Check the unit column.
- **The preservation trio drives the answer more than anything else.** Moving
  `fi_preservation_ratio` from `0.5` to `0.75`, or raising a confidence from
  `0.5` to `0.9`, can change the required capital several-fold. When a result
  looks surprising, vary these one at a time before concluding anything.
- **`withdrawal_percentage_stock` is a yardstick, not a lever.** It changes how
  the income requirement is *sized*, not what the simulation finds sustainable.
  Raising it makes the requirement look smaller without changing any outcome.
- **Structural parameters are intentionally not exposed** — tax rates,
  `basiszins`, `teilfreistellung`, simulation counts. Changing them silently
  produces confident but meaningless comparisons.
- **Compare like with like**: two scenarios are only comparable if they differ
  in the parameters under study and share everything else, including the seed.

### Roadmap

Structured `--json` output, a reduced-simulation `--fast` mode, section
selection and explicit `--record` gating of run-history side effects are in
progress; they complete the MCP tool surface
(`run_plan_check`, `get_last_summary`, `get_run_history`).

---

## Disclaimer

A personal modelling tool, not financial advice. Outputs are conditional on
assumptions that are documented, debatable, and sometimes wrong — which is the
point of `METHODOLOGY.md`.

## License

MIT
