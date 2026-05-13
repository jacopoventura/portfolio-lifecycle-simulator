import time
import anthropic
from dotenv import load_dotenv

load_dotenv()

_SYSTEM_PROMPT = """\
You are a personal finance advisor reviewing a Monte Carlo retirement planning simulation.
You receive three inputs:
  1. The user's config (inputs and assumptions)
  2. The full detailed simulation report (all grids, sensitivity tables, MC scans)
  3. A clean Q1–Q4 summary of the key results

Answer exactly these 5 questions, one bullet each, in this order:

  Q1. What is the target portfolio (stocks + bonds) required at retirement, and is the
      current plan on track to reach it?

  Q2. Which combination of monthly stock and bond contributions achieves the target
      portfolio (stocks ≥ required AND bonds ≥ required) with the highest success rate?
      Give the best stock/mo + bond/mo pair and its success rate.

  Q3. Which combination of monthly stock and bond contributions achieves the target
      retirement income with the highest success rate (any portfolio split)?
      Give the best stock/mo + bond/mo pair and its success rate.

  Q4. If the current contribution plan is kept unchanged, when is financial freedom
      reached at the required success rate, regardless of portfolio composition?

  Q5. What is the optimal cash reallocation today (how much to move into stocks and/or
      bonds) to maximise the probability of reaching the retirement income target?

  Q6. Combining all findings above (contributions, cash reallocation, CAGR assumptions),
      what is the single most actionable recommendation to reach financial freedom in the
      IT scenario as early as possible at the required success rate? Be specific: state
      the recommended monthly contributions, any upfront cash move, and the resulting
      freedom year and success rate.

  Q7. Based on the sensitivity waterfall table in the report, identify the top 2 risks
      to the IT retirement plan. For each risk: name it, state its impact in pp, explain
      why it matters, and suggest one concrete mitigation action.

  Q8. Flag the top 3 things in this report that deserve scrutiny before acting on the
      results. For each: name the issue, explain why it is suspicious or potentially
      misleading, and state what assumption change or additional check would resolve it.
      Focus on model inputs, parameter choices, or results that look extreme or
      inconsistent with each other.

Priority rules:
- IT (Italy) scenario is the primary residence — always lead with IT numbers.
  Mention DE only when it differs meaningfully.
- Never invent numbers — every figure must come directly from the report.
- Flag explicitly if the 97% success target is not achievable.
- Express all monetary values in full euros or thousands (e.g. €2,716k) — never billions.

Output format — exactly 8 bullet points, one per question:
- Label: **Q1 —**, **Q2 —**, **Q3 —**, **Q4 —**, **Q5 —**, **Q6 —**, **Q7 —**, **Q8 —**
- Start with a bold action verb after the label
- At least one specific number per bullet
- Maximum 45 words per bullet (Q6, Q7, and Q8 may use up to 80 words)
- No preamble, no headers, no closing sentence
"""


_DEFAULT_MODEL = "claude-sonnet-4-6"
_FALLBACK_MODEL = "claude-haiku-4-5-20251001"

_UNIFIED_SYSTEM_PROMPT = """\
You are a personal finance advisor reviewing a full retirement planning analysis:
  1. Accumulation Monte Carlo — reaching the target portfolio by retirement
  2. Decumulation Monte Carlo — whether that portfolio survives 40 years of withdrawals
  3. CAGR / instrument analysis — which ETFs credibly deliver the required stock return

The required stock CAGR (minimum return to hit the target at the required success rate) is
provided as a number. Use it as the CAGR benchmark in Q4 and Q6.

Answer exactly these 6 questions, one bullet each:

  Q1. Required portfolio & status: state the required portfolio at retirement, whether the
      current plan reaches it, and the financial freedom year at the required success rate.

  Q2. What to change: give the single best monthly stock + bond contribution pair and the
      optimal upfront cash reallocation (stocks/bonds split) to hit the income target as
      early as possible. State the resulting freedom year and success rate.
      Then explicitly reconcile the two tables in the report: the "EXACT SPLIT" table shows
      the monthly-only path (no lump sum), which requires high monthly stock contributions
      because the cash pile is left idle; the "BEST CASH ALLOCATION TODAY" table deploys
      existing cash upfront as a lump sum, so the required monthly contributions drop
      significantly. Explain this trade-off in one sentence so the user is not confused.

  Q3. Survival: does the target portfolio survive 40 years of withdrawals? State the joint
      P(accumulate AND survive). If below the target success rate, state what fixes it
      (higher contributions, lower SWR, or extended horizon).

  Q4. Instruments: propose 2 ETF portfolio combinations whose P5-10y CAGR meets or exceeds
      the required stock CAGR. Each combination: 2–4 instruments, low correlation, MDD within
      range. Format each as "Name (TICKER) — weight%  |  P5 CAGR  |  MDD". QQQ ≤ 25%.
      Prefer EU-eligible UCITS ETFs (✓) unless a US instrument adds meaningful diversification.

  Q5. Top risks: name the 2 biggest risks to the plan, state each impact in percentage
      points, and give one concrete mitigation per risk.

  Q6. Action plan: one concrete recommendation combining contributions, cash move, best
      instrument option from Q4, and decumulation check. State freedom year and success rate.

Rules:
- IT (Italy) is the primary scenario — lead with IT numbers; mention DE only if it differs.
- Never invent numbers — every figure must come from the reports.
- Flag explicitly if the target success rate is not achievable.
- Monetary values in euros or thousands (e.g. €2,716k) — never billions.
- ETF names always as "Full Name (TICKER)".

Output format — exactly 6 bullet points, labels **Q1 —** through **Q6 —**:
- Bold action verb after each label
- At least one specific number per bullet
- Q1, Q3, Q5: max 50 words. Q2: max 90 words. Q4: max 120 words. Q6: max 70 words.
- No preamble, no headers, no closing sentence.
"""


def generate_action_plan(
    config_toml: str,
    verbose_report: str,
    clean_summary: str,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 1400,
) -> str:
    user_message = (
        "## Configuration\n\n"
        + config_toml
        + "\n\n## Full Detailed Report\n\n"
        + verbose_report
        + "\n\n## Clean Summary\n\n"
        + clean_summary
    )
    return _api_call(_SYSTEM_PROMPT, user_message, model, max_tokens)


def _api_call(system: str, user_message: str, model: str, max_tokens: int) -> str:
    client = anthropic.Anthropic()
    _MAX_RETRIES = 4
    _BASE_DELAY  = 10  # seconds

    for attempt in range(_MAX_RETRIES):
        try:
            msg = client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user_message}],
            )
            break
        except anthropic.NotFoundError:
            print(f"\n  [Claude API] Model '{model}' not found, falling back to {_FALLBACK_MODEL}...")
            model = _FALLBACK_MODEL
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < _MAX_RETRIES - 1:
                delay = _BASE_DELAY * (2 ** attempt)
                print(f"\n  [Claude API] Overloaded (529) — retrying in {delay}s "
                      f"(attempt {attempt + 1}/{_MAX_RETRIES})...")
                time.sleep(delay)
            else:
                raise
    else:
        raise RuntimeError("Claude API overloaded after all retries — try again in a few minutes.")

    tokens_in  = msg.usage.input_tokens
    tokens_out = msg.usage.output_tokens
    cost = tokens_in * 0.00000025 + tokens_out * 0.00000125
    print(f"\n[Claude API] tokens: {tokens_in} in / {tokens_out} out — est. cost: ${cost:.4f}")
    print("[Claude API] Check remaining credits at: console.anthropic.com/settings/billing\n")
    return msg.content[0].text


def generate_unified_action_plan(
    config_toml: str,
    accum_verbose: str,
    accum_summary: str,
    decum_report: str,
    cagr_agent_text: str,
    required_cagr_it: float | None = None,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 1500,
) -> str:
    """Unified AI action plan covering accumulation, decumulation and instrument selection."""
    cagr_note = (
        f"\n\nRequired stock CAGR (from accumulation, IT scenario): "
        f"{required_cagr_it * 100:.2f}%"
        if required_cagr_it is not None else ""
    )
    user_message = (
        "## Configuration\n\n" + config_toml
        + cagr_note
        + "\n\n## Accumulation Report (Full)\n\n" + accum_verbose
        + "\n\n## Accumulation Summary\n\n" + accum_summary
        + "\n\n## Decumulation Report\n\n" + decum_report
        + "\n\n## CAGR / Instrument Analysis\n\n" + cagr_agent_text
    )
    return _api_call(_UNIFIED_SYSTEM_PROMPT, user_message, model, max_tokens)
