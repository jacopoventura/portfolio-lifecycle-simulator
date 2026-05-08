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


def generate_action_plan(
    config_toml: str,
    verbose_report: str,
    clean_summary: str,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 1400,
) -> str:
    client = anthropic.Anthropic()
    user_message = (
        "## Configuration\n\n"
        + config_toml
        + "\n\n## Full Detailed Report\n\n"
        + verbose_report
        + "\n\n## Clean Summary\n\n"
        + clean_summary
    )
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.NotFoundError:
        print(f"\n  [Claude API] Model '{model}' not found, falling back to {_FALLBACK_MODEL}...")
        msg = client.messages.create(
            model=_FALLBACK_MODEL,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    input_tokens = msg.usage.input_tokens
    output_tokens = msg.usage.output_tokens
    cost = input_tokens * 0.00000025 + output_tokens * 0.00000125
    print(f"\n[Claude API] tokens: {input_tokens} in / {output_tokens} out — est. cost: ${cost:.4f}")
    print("[Claude API] Check remaining credits at: console.anthropic.com/settings/billing\n")
    return msg.content[0].text
