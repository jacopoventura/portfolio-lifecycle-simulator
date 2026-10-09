# portfolio-lifecycle-simulator

[![tests](https://github.com/jacopoventura/portfolio-lifecycle-simulator/actions/workflows/tests.yml/badge.svg)](https://github.com/jacopoventura/portfolio-lifecycle-simulator/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

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

## Example output

Abridged from a real run on template figures — see `config.example.toml`.
Generated before the October 2026 model update (CAGR-consistent stock returns,
persistent bond yields, annual rebalancing); current runs show higher
survival/preservation and a lower required wealth. See `METHODOLOGY.md` §8.

### S1 · Accumulation

```text
THREE WEALTH NUMBERS BELOW — don't confuse them (all at 2041 prices; today's € in parentheses):
    €1.95M   income min        covers the required net €4,219/mo at 2041 (€3,000 today) with 50/50 stocks/bonds
    €1.49M   plan P50 (MC)     generates ≈€3,231/mo (≈€2,297 today) with 65/35 stocks/bonds
    €4.19M   preservation min  keeps ≥60% real wealth to 2071 (€2.98M today)

Required portfolio at horizon (15y)
     Target income : 4,219/mo  (50,633/yr, inflation-adjusted)
     Composition   : stocks 50.0% · bonds 50.0%
     DE :      2,076,483  (stocks 1,038,241 · bonds 1,038,241)
     IT :      1,946,587  (stocks 973,293 · bonds 973,293)

  2. Contribution needed to reach the TARGET INCOME with ≥95% confidence
     (fixed CAGR of 6.0%, current allocation kept)
        DE : stocks 4,750/mo · bonds 6,024/mo · total 10,774/mo   P(income)=✓95.3%
        IT : stocks 4,500/mo · bonds 5,522/mo · total 10,022/mo   P(income)=✓95.0%

  3. Major risks  (top 2 negative impacts on income success)
     DE : Inflation (+δ ±1pp) → -7.7pp  ·  Stock CAGR (−δ ±2pp) → -7.1pp
     IT : Inflation (+δ ±1pp) → -9.1pp  ·  Stock CAGR (−δ ±2pp) → -7.6pp
```

### S2 · Survival under stress

```text
At the plan's current allocation, covering €4,219/mo nominal at retirement.
  Each cell: SURVIVAL % (median capital PRESERVED %, real €) — see below the table.

      Portfolio                                      Baseline      Crash yr1-5        2000-2002        2008-2009
      ──────────────────────────────────────────────────────────────────────────────────────────────────────────
      €1.95M  IT required portfolio                 98% (59%)        90% (24%)        85% (22%)        96% (39%)
      €1.76M  10% preservation                      95% (48%)        76% (13%)        71% (11%)        88% (28%)
      €2.08M  25% preservation                      99% (65%)        95% (30%)        92% (28%)        98% (45%)
      €3.20M  50% preservation                     100% (94%)       100% (61%)       100% (58%)       100% (76%)
      €4.23M  60% preservation (configured target) 100% (107%)       100% (74%)       100% (71%)       100% (89%)

  How to read this table
    Each ROW is a candidate starting portfolio. Each COLUMN is a market
    condition it's tested against: "Baseline" = ordinary random returns;
    the other three force a real historical crash into the first years of
    retirement. Each cell shows two numbers:
      SURVIVAL  — P(the portfolio never fully runs out over 30 years)
      PRESERVED — the typical (median) real capital left at the end, as a
                  % of what you started with (in parentheses)
```

### S3 · Instruments

```text
Real-world CAGR distributions  (P5 = bad-luck entry, P50 = typical):

     Source                  Period             P5     P50
     ──────────────────────  ─────────────  ──────  ──────
     S&P 500  USD            2000–2026       +6.3%   +6.5%
     S&P 500  USD            2000–2015       +2.0%   +2.3%
     S&P 500  EUR            2003–2026       +9.0%   +9.3%
     DCA      historical     2000–2015       +4.7%   +4.9%
     Lump-sum 2-phase MC     2000–2015       -3.4%   +2.1%
     DCA      2-phase MC     2000–2015       +0.3%   +4.4%


 Common window  2016-05 → 2026-07   10.2y
    
         k  Allocation                                  P50     P5  Sharpe  Sortino     MDD  Avg corr
      ─────────────────────────────────────────────────────────────────────────────────────────
        ★2  QQQ 40% + TDIV 60%                        16.3%  15.9%   1.07   1.31  -32.2%   0.32
        ☆3  QQQ 40% + TDIV 60%                        16.3%  15.9%   1.07   1.31  -32.2%   0.32
         7  IS3R 23% + QQQ 20% + TDIV 57%             15.0%  14.7%   1.05   1.26  -32.8%   0.47
       all  IS3R 23% + QQQ 20% + TDIV 57%             15.0%  14.7%   1.05   1.26  -32.8%   0.47
         6  IS3R 23% + QQQ 20% + TDIV 57%             15.0%  14.7%   1.05   1.26  -32.8%   0.47
         5  IS3R 22% + QQQ 20% + TDIV 57%             15.0%  14.7%   1.05   1.26  -32.8%   0.47
         4  IS3R 22% + QQQ 20% + TDIV 58%             15.0%  14.7%   1.05   1.26  -32.8%   0.47
      ─────────────────────────────────────────────────────────────────────────────────────────
       ref  S&P 500  etf-window  [2016–2026] (EUR)    13.1%  12.7%      —      —  -33.3%      —
       ref  S&P 500  2000–today  [2000–2026]           6.4%  -1.5%      —      —  -56.8%      —
```

---

## Modelling

- **Stochastic returns** — 50,000 paths; Student-t innovations for fat tails,
  AR(1) inflation. The stock input is a **CAGR**: paths are generated so their
  median compounded growth equals it (no hidden σ²/2 volatility drag).
- **Bonds** — persistent, inflation-linked yields (rate regimes last years;
  yields rise with inflation), coupon plus duration-based price changes, and a
  stock–bond correlation defined on returns (negative = bonds hedge).
- **Annual rebalancing** to the target mix (or along the glide path).
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
- **External validation** — the decumulation engine is tested against the
  Trinity study and Bengen: 4% / 50-50 / 30 years gives ≈ 93% survival vs the
  published ≈ 95%. Details, and the remaining known limitation, in
  `METHODOLOGY.md` §8.

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
| `expected_yearly_growth_stocks_percent` | % per year, e.g. `6` | Planning CAGR for equities (median compounded growth), net of fund costs. Nominal, not real. The engine converts it to the arithmetic mean that compounds at this rate |
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

### MCP server

`financial_planning_server.py` runs this engine as a stdio MCP server. A
sibling repo, `AIagent` (a personal chat agent for career + financial
planning), spawns it as one of its tools — see that repo's `TODO.md` Phase 4
for the cross-repo architecture. It reuses the same override/isolation
machinery described above (`apply_cli_overrides` + `FIRE_CONFIG`), so a call
from the agent gets the same "never touches the real `config.toml`"
guarantee a human running `main.py --target-monthly-retirement-income 3000`
gets.

One tool is exposed today: `accumulation_projection` — the S1 accumulation
question (required portfolio, current success probability, required CAGR)
for the IT tax scenario, with the full text report available on request via
`include_report=True`. More tools are planned, wrapping `monte_carlo.py`'s
pure compute functions directly (decumulation preservation, two-bucket mode,
survival matrices) rather than going through this CLI-flag surface.

---

## Lessons learned — serving an engine over MCP

Non-obvious things that cost real debugging time when wrapping this engine as
an MCP server. The client-side counterparts live in
[AIagent](https://github.com/jacopoventura/AIagent)'s README.

**stdout belongs to the protocol.** A stdio MCP server speaks JSON-RPC over
stdout. This engine prints progress markers as it runs, and every one of them
would corrupt the stream — silently, producing confusing frame errors rather
than an obvious crash. Every call into the pipeline from the server is wrapped
in `contextlib.redirect_stdout`.

**Import the engine; don't shell out to it.** The obvious design — subprocess
`main.py`, parse the output — would have required structured `--json` output, a
`--fast` flag, and gating the run-history side effects. None of that was
necessary. `run_accum`, `run_decum` and `run_cagr` are plain importable
functions, and every file write (`run_history.csv`, `drift_history.*`,
`last_run_*`) lives in `main()` or `_save_context()` — not in them. So the
server calls the functions directly, gets typed results with nothing to parse,
and writes nothing. Checking where the side effects actually lived removed the
entire blocker list.

**A server called from another repo needs three things set explicitly.** The
client spawns this server as a subprocess, and naive spawning fails three ways:
the wrong interpreter (the client's venv has no numpy/pandas/scipy), the wrong
`sys.path` (Python adds the *script's* directory — which is why this server
sits at the repo root, so `import main` resolves), and the wrong working
directory (`resolve_config_path()` returns a *relative* path). All three are
configuration on the client side; `StdioServerParameters` accepts `command`,
`cwd` and `env`.

**Tool descriptions are prompt engineering, not documentation.** Asked what
portfolio covers a given monthly income, the agent initially asked four
clarifying questions and then fell back to the 4% rule — a figure ~30% below
what this engine computes at the user's configured 2.8% withdrawal rate. It had
the tool and didn't call it, because nothing told it that omitted parameters
fall back to the saved plan. The fix was entirely in text the model reads:
state the fallback explicitly, give every parameter a `Field(description=...)`
with units and ranges, **interpolate the live config values into the tool
description at server startup**, and return `assumptions_used` alongside the
result so an answer can say which values were applied rather than applying them
silently.

**Timings, measured** (2,000 simulations, this repo's own functions):
`simulate_decumulation` 0.04s, `compute_canonical_fi_required_scan` 5.5s,
full `run_accum` + `run_decum` pipeline ~17s. Latency scales with simulation
count and, for the scan, with how many candidate wealth values the bisection
evaluates. Fast enough for synchronous tool calls; a job/polling contract was
considered and rejected on this evidence rather than assumed.

---

## Disclaimer

A personal modelling tool, not financial advice. Outputs are conditional on
assumptions that are documented, debatable, and sometimes wrong — which is the
point of `METHODOLOGY.md`.

## License

MIT
