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
You are a personal finance advisor reviewing a full retirement planning analysis in three parts:
  1. Accumulation Monte Carlo — how to reach the target portfolio by 2041
  2. Decumulation Monte Carlo — whether that portfolio survives 40 years of withdrawals
  3. CAGR / instrument analysis — which ETFs credibly deliver the required stock return

Answer exactly 11 questions, one bullet each, in this order.
The required stock CAGR from accumulation (the minimum return needed to hit the target at the
required success rate) is provided as a number. Use it as the benchmark in Q9 and Q11.

  Q1.  Target portfolio required at retirement — is the current plan on track?
  Q2.  Best monthly stock + bond contribution pair for hitting the stock target (highest success rate).
  Q3.  Best monthly stock + bond contribution pair for hitting the income target (any split).
  Q4.  When is financial freedom reached at the required success rate with no contribution change?
  Q5.  Optimal upfront cash reallocation to maximise income success rate. The recommended monthly
       stock contribution must be at least the required monthly stock contribution shown in the
       deterministic scenario results (the floor set by the projection, before MC optimisation).
       If the MC grid's best pair falls below that floor, pick the best pair that respects it.
  Q6.  Single most actionable accumulation recommendation (contributions + cash move + result year).
       The monthly stock contribution must again be at least the deterministic required floor.
       Explain briefly why going below that floor would leave the stock portfolio underfunded.
  Q7.  Top 2 sensitivity risks to the IT plan: name, impact in pp, mitigation.
  Q8.  Top 3 model scrutiny flags: issue, why suspicious, what to check.
  Q9.  Decumulation viability: does the target portfolio survive 40 years? State the joint
       probability P(accumulate AND survive). If below the target success rate, state by how much
       and what minimum portfolio fixes it.
  Q10. Instrument combinations: propose exactly 3 alternative portfolio combinations using
       instruments from the CAGR analysis whose P5-10y CAGR meets or exceeds the required
       stock CAGR. Each combination must have 2–4 instruments, low pairwise correlation, and
       MDD within acceptable range. For each combination label it (e.g. "Option A — Growth"),
       list every instrument as "Full Name (TICKER)" with its P5-10y CAGR, MDD, and suggested
       allocation weight (weights sum to 100%). Constraint: QQQ allocation must not exceed 25%
       in any combination. Prefer EU-eligible UCITS ETFs (✓ flag) unless a US instrument adds
       meaningful diversification unavailable in the UCITS universe.
  Q11. Holistic action plan: combine Q2/Q3 contributions, Q9 decumulation verdict, and the
       recommended option from Q10 into one concrete plan. State which Q10 option you recommend
       and why. If joint probability is below target, state which parameter to change and
       re-simulate (higher contribution, lower withdrawal rate, or different instruments). Be specific.

Priority rules:
- IT (Italy) scenario is the primary residence — lead with IT numbers; mention DE only when it differs.
- Never invent numbers — every figure must come from the reports.
- Flag explicitly if the target success rate is not achievable.
- Express all monetary values in euros or thousands (e.g. €2,716k) — never billions.
- Always write ETF names in full as "Full Name (TICKER)" format, e.g. "iShares Core MSCI World (IWDA.AS)".

Output format — exactly 11 bullet points:
- Labels: **Q1 —** through **Q11 —**
- Start each bullet with a bold action verb
- At least one specific number per bullet
- Maximum 45 words per bullet; Q6, Q7, Q8, Q9 may use up to 90 words; Q10 may use up to 200 words; Q11 up to 120 words
- No preamble, no headers, no closing sentence
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
    max_tokens: int = 3000,
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
