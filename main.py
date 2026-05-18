"""
main.py — unified retirement planning analysis

Runs the full four-section pipeline in sequence:

  1. ACCUMULATION  — what portfolio/contributions do I need to retire by 2041?
  2. DECUMULATION  — can that portfolio survive 40 years of withdrawals?
  3. INSTRUMENTS   — which ETFs credibly deliver the required stock return?
  4. AI AGENT      — action plan integrating all three sections (opt-in via --agent)

Usage:
  python main.py                     # sections 1→2→3 (no AI call)
  python main.py --agent             # full pipeline including AI agent (section 4)
  python main.py --from decum        # start from decumulation (uses config values for p50/tail)
  python main.py --from cagr         # CAGR only
  python main.py --from agent        # re-run agent on last captured context
  python main.py --verbose           # pass --verbose to accumulation section

Requires ANTHROPIC_API_KEY in .env for section 4.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

from models import DecumulationSummary, PlanVerdictSummary


# ── Helpers ───────────────────────────────────────────────────────────────────

_CONTEXT_FILE = Path("historical_data") / "last_run_context.txt"


def _fmt_eur(v: float | None) -> str:
    return f"€{v:,.0f}" if v is not None else "—"


def _fmt_pct(v: float | None, scale: float = 100.0) -> str:
    return f"{v * scale:.1f}%" if v is not None else "—"


def print_plan_snapshot(
    plan_verdict: PlanVerdictSummary | None,
    decum_summary: DecumulationSummary | None,
    required_cagr: float | None,
    p50: float,
) -> None:
    W = 70
    print()
    print(f"{'  PLAN SNAPSHOT':^{W}}")
    print()

    print(f"  Accumulation")
    if plan_verdict is not None:
        req_portf   = _fmt_eur(plan_verdict.required_portfolio)
        proj_portf  = _fmt_eur(plan_verdict.projected_portfolio_config)
        gap         = plan_verdict.gap_config
        gap_sign    = "+" if gap >= 0 else "−"
        gap_str     = f"{gap_sign}€{abs(gap):,.0f}"
        on_track    = "✓  ON TRACK" if plan_verdict.on_track else "✗  SHORTFALL"
    else:
        req_portf = proj_portf = gap_str = "—"
        on_track  = "—"

    req_cagr_str = _fmt_pct(required_cagr)
    print(f"    Required portfolio   : {req_portf}")
    print(f"    Projected (config)   : {proj_portf}   gap {gap_str}")
    if p50 > 0:
        print(f"    MC P50 at horizon    : €{p50:,.0f}")
    print(f"    Required stock CAGR  : {req_cagr_str}  (IT scenario)")
    print(f"    Verdict              : {on_track}")
    print()

    # Prefer the joint rate reported in the CAGR Plan Verdict (chained MC sample
    # the cagr section uses); fall back to decum's chained joint if not present.
    joint_acc_surv: float | None = None
    if plan_verdict is not None and plan_verdict.joint_survival_rate is not None:
        joint_acc_surv = plan_verdict.joint_survival_rate
    elif decum_summary is not None:
        joint_acc_surv = decum_summary.joint_survival_chained

    if decum_summary is not None:
        req_decum   = _fmt_eur(decum_summary.required_portfolio_income)
        target_wd   = _fmt_eur(decum_summary.target_monthly_withdrawal)
        surv_fixed  = _fmt_pct(decum_summary.survival_fixed_final)
        surv_gk     = _fmt_pct(decum_summary.survival_gk_final)
        horizon_yr  = decum_summary.n_years
    else:
        req_decum = target_wd = surv_fixed = surv_gk = "—"
        horizon_yr = 40

    print(f"  Decumulation  ({horizon_yr}yr horizon, fixed nominal withdrawal)")
    print(f"    Required portfolio   : {req_decum}  →  {target_wd}/mo")
    print(f"    Survival yr{horizon_yr} — fixed: {surv_fixed}")
    print(f"    Survival yr{horizon_yr} — GK   : {surv_gk}")
    print(f"    Joint P(acc ∧ surv)  : {_fmt_pct(joint_acc_surv)}")
    print()
    print()

_SECTIONS = ("accum", "decum", "cagr", "agent")


def _banner(n: int, title: str) -> None:
    print()
    print("─" * 80)
    print(f"SECTION {n}  —  {title}")
    print("─" * 80)


class _Tee:
    """Write to both real stdout and a capture buffer simultaneously."""
    def __init__(self, stream, buf: io.StringIO) -> None:
        self._stream = stream
        self._buf = buf

    def write(self, s: str) -> int:
        self._stream.write(s)
        self._stream.flush()
        self._buf.write(s)
        return len(s)

    def flush(self) -> None:
        self._stream.flush()


def _capture(fn, argv_extra: list[str] | None = None):
    """Call fn() with a clean sys.argv, print output live, and return (text, return_value)."""
    buf = io.StringIO()
    real_stdout = sys.stdout          # save before redirect_stdout replaces it
    tee = _Tee(real_stdout, buf)
    saved = sys.argv
    sys.argv = [saved[0]] + (argv_extra or [])
    try:
        with contextlib.redirect_stdout(tee):
            result = fn()
    finally:
        sys.argv = saved
    return buf.getvalue(), result


_SUMMARIES_FILE = _CONTEXT_FILE.with_name("last_run_summaries.json")


def _save_context(
    accum: str, decum: str, cagr: str,
    required_cagr: float | None,
    decum_summary: DecumulationSummary | None,
    plan_verdict: PlanVerdictSummary | None,
) -> None:
    import json
    from dataclasses import asdict
    _CONTEXT_FILE.parent.mkdir(exist_ok=True)
    with _CONTEXT_FILE.open("w") as f:
        f.write(f"REQUIRED_CAGR_IT={required_cagr or ''}\n")
        f.write("===ACCUM===\n" + accum)
        f.write("\n===DECUM===\n" + decum)
        f.write("\n===CAGR===\n" + cagr)
    payload = {
        "decum_summary": asdict(decum_summary) if decum_summary else None,
        "plan_verdict":  asdict(plan_verdict)  if plan_verdict  else None,
    }
    _SUMMARIES_FILE.write_text(json.dumps(payload, indent=2))


def _load_context() -> tuple[str, str, str, float | None, DecumulationSummary | None, PlanVerdictSummary | None]:
    if not _CONTEXT_FILE.exists():
        return "", "", "", None, None, None
    text = _CONTEXT_FILE.read_text()
    required_cagr = None
    if text.startswith("REQUIRED_CAGR_IT="):
        first_line, text = text.split("\n", 1)
        val = first_line.split("=", 1)[1].strip()
        required_cagr = float(val) if val else None
    accum = text.split("===DECUM===")[0].replace("===ACCUM===\n", "")
    decum = text.split("===DECUM===")[1].split("===CAGR===")[0].strip()
    cagr  = text.split("===CAGR===")[1].strip() if "===CAGR===" in text else ""

    decum_summary: DecumulationSummary | None = None
    plan_verdict:  PlanVerdictSummary  | None = None
    if _SUMMARIES_FILE.exists():
        import json
        payload = json.loads(_SUMMARIES_FILE.read_text())
        if payload.get("decum_summary"):
            decum_summary = DecumulationSummary(**payload["decum_summary"])
        if payload.get("plan_verdict"):
            plan_verdict = PlanVerdictSummary(**payload["plan_verdict"])
    return accum, decum, cagr, required_cagr, decum_summary, plan_verdict


# ── Sections ──────────────────────────────────────────────────────────────────

def run_accum(verbose: bool = False) -> tuple[str, float, float, float | None, int]:
    """Run accumulation section. Returns (report_text, p50, tail, required_cagr, tail_pct)."""
    import main_portfolio_projection as m

    _banner(1, "ACCUMULATION — target portfolio, contributions, financial freedom")
    argv = ["--no-clear"]
    if verbose:
        argv.append("--verbose")
    text, result = _capture(m.main, argv)

    if result is None:
        print("\n  [!] Accumulation failed — check config.toml")
        return text, 0.0, 0.0, None, 3
    p50, tail, required_cagr, tail_pct, _vr, _cs, _cfg = result
    return text, p50, tail, required_cagr, tail_pct


def run_decum(p50: float, tail: float, tail_pct: int) -> tuple[str, DecumulationSummary | None]:
    """Run decumulation section. Returns (report text, typed summary)."""
    import main_decumulation as m

    print()
    _banner(2, "DECUMULATION — 40-year withdrawal survival & joint probability")
    text, summary = _capture(
        lambda: m.main(
            p50_override=p50 if p50 > 0 else None,
            tail_override=tail if tail > 0 else None,
        ),
        ["--no-clear", "--tail-pct", str(tail_pct)],
    )
    return text, summary if isinstance(summary, DecumulationSummary) else None


def run_cagr() -> tuple[str, object, PlanVerdictSummary | None]:
    """Run CAGR / instrument section.

    Returns (agent_summary_text, OptimalPortfolioResult | None, PlanVerdictSummary | None).
    The text is trimmed to the AGENT SUMMARY section to keep agent context compact.
    """
    import main_cagr_analysis as m

    _banner(3, "INSTRUMENTS — historical CAGR, volatility, correlation + optimiser")
    text, result = _capture(m.main, ["--no-clear"])

    # result is (list[TickerResult], OptimalPortfolioResult | None, PlanVerdictSummary | None)
    opt_result = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    plan_verdict = result[2] if isinstance(result, tuple) and len(result) > 2 else None

    marker = "AGENT SUMMARY"
    agent_text = text[text.index(marker):] if marker in text else text
    return agent_text, opt_result, plan_verdict


def run_agent(
    accum_text: str,
    decum_text: str,
    cagr_text: str,
    required_cagr: float | None,
    config_toml: str,
) -> None:
    """Call the unified AI agent with all three section outputs."""
    _banner(4, "AI AGENT — action plan")
    try:
        from agent import generate_unified_action_plan
    except ImportError:
        print("  Install the 'anthropic' package to use the AI agent.")
        return

    print("  Generating unified action plan via Claude Sonnet…")
    print()
    try:
        plan = generate_unified_action_plan(
            config_toml=config_toml,
            accum_verbose=accum_text,
            accum_summary="",
            decum_report=decum_text,
            cagr_agent_text=cagr_text,
            required_cagr_it=required_cagr,
        )
        print(plan)
        print()
    except Exception as e:
        print(f"  [!] Agent call failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified retirement planning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--from", dest="start_from", choices=_SECTIONS, default="accum",
        metavar="SECTION",
        help="start from this section (accum | decum | cagr | agent). "
             "Sections before the start point use cached values from last run.",
    )
    parser.add_argument("--agent", action="store_true",
                        help="run AI agent (section 4) after sections 1–3")
    parser.add_argument("--verbose", action="store_true",
                        help="pass --verbose to accumulation section")
    args = parser.parse_args()

    start = _SECTIONS.index(args.start_from)

    # Load cached context for sections we're skipping
    accum_text, decum_text, cagr_text, required_cagr, decum_summary, plan_verdict = _load_context()
    p50 = tail = 0.0
    tail_pct = 3
    config_toml = Path("config.toml").read_text() if Path("config.toml").exists() else ""

    # Section 1: Accumulation
    if start <= _SECTIONS.index("accum"):
        accum_text, p50, tail, required_cagr, tail_pct = run_accum(verbose=args.verbose)

    # Section 2: Decumulation
    if start <= _SECTIONS.index("decum"):
        decum_text, decum_summary = run_decum(p50, tail, tail_pct)

    # Section 3: CAGR + optimiser
    if start <= _SECTIONS.index("cagr"):
        cagr_text, _opt, plan_verdict = run_cagr()

    # Cache context for --from agent re-runs
    _save_context(accum_text, decum_text, cagr_text, required_cagr, decum_summary, plan_verdict)

    # Plan snapshot — compact summary printed before agent
    print_plan_snapshot(plan_verdict, decum_summary, required_cagr, p50)

    # Section 4: AI Agent — only when explicitly requested
    if args.agent or args.start_from == "agent":
        run_agent(accum_text, decum_text, cagr_text, required_cagr, config_toml)


if __name__ == "__main__":
    main()
