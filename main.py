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

from models import CagrRealismSnapshot, CanonicalFIRequiredScan, DecumulationSummary, PlanVerdictSummary


# ── Helpers ───────────────────────────────────────────────────────────────────

_CONTEXT_FILE = Path("historical_data") / "last_run_context.txt"
_HISTORY_FILE = Path("historical_data") / "run_history.csv"
_HISTORY_HEADER = "date,required_cagr,joint_P,p50\n"

_DRIFT_FILE = Path("historical_data") / "drift_history.csv"
_DRIFT_HEADER = (
    "month,bonds_pct,horizon_years,target_income_eur,"
    "target_portfolio_it_eur,success_rate_it_bonds,success_rate_it_stocks,"
    "success_rate_it_total,lever_a_monthly_stocks,lever_a_monthly_bonds\n"
)
# Index of "success_rate_it_total" in a CSV row (0-based).
_DRIFT_TOTAL_COL = 7


def _append_history(
    required_cagr: float | None,
    joint_P: float | None,
    p50: float,
) -> None:
    """Append one row to run_history.csv so drift over time is visible.
    Values stored as decimals (e.g. 0.042 for 4.2%); empty string when None."""
    from datetime import date
    _HISTORY_FILE.parent.mkdir(exist_ok=True)
    fresh = not _HISTORY_FILE.exists()
    def _fmt(v: float | None, ndigits: int) -> str:
        return f"{v:.{ndigits}f}" if v is not None else ""
    row = (
        f"{date.today().isoformat()},"
        f"{_fmt(required_cagr, 5)},"
        f"{_fmt(joint_P, 4)},"
        f"{_fmt(p50 if p50 > 0 else None, 0)}\n"
    )
    with _HISTORY_FILE.open("a") as f:
        if fresh:
            f.write(_HISTORY_HEADER)
        f.write(row)


def _lever_a_monthly_split(verdict_inputs) -> tuple[float, float]:
    """Recompute Lever A monthly stock + bond contributions for the IT scenario.
    Mirrors print_final_verdict so drift tracking works even when the verdict
    block is suppressed. Returns (stocks_per_mo, bonds_per_mo)."""
    from reports.section2 import _find_min_total_income_split
    from reports.summary import _closest_cagr_idx
    config = verdict_inputs.config
    assumed = config.expected_yearly_growth_stocks_percent / 100
    cagr_idx = _closest_cagr_idx(verdict_inputs.grids_it[0].cagr_values, assumed)
    stocks, bonds, _ = _find_min_total_income_split(
        verdict_inputs.grids_it, cagr_idx, verdict_inputs.target_success,
    )
    return stocks, bonds


def _append_drift_snapshot(
    verdict_inputs,
    accum_required_it: float,
) -> None:
    """Save a per-month snapshot of plan state + simulation outcome.

    One row per calendar month (MM-YYYY). When the same month already has a
    row, we overwrite it ONLY if the new IT income success rate beats what's
    on file — so the log becomes "best snapshot so far this month" rather
    than a noisy per-run dump."""
    from datetime import date

    config = verdict_inputs.config
    mc = verdict_inputs.mc_it
    a_stocks, a_bonds = _lever_a_monthly_split(verdict_inputs)

    month = date.today().strftime("%m-%Y")
    sr_total = mc.success_rate_income_only  # IT income success — the headline drift metric
    row = (
        f"{month},{config.bonds_percentage:.2f},{int(round(config.years))},"
        f"{config.target_monthly_retirement_income:.2f},"
        f"{accum_required_it:.0f},"
        f"{mc.success_rate_bond_only:.4f},{mc.success_rate_stock_only:.4f},"
        f"{sr_total:.4f},{a_stocks:.2f},{a_bonds:.2f}\n"
    )

    _DRIFT_FILE.parent.mkdir(exist_ok=True)
    header = _DRIFT_HEADER
    existing: list[str] = []
    if _DRIFT_FILE.exists():
        lines = _DRIFT_FILE.read_text().splitlines(keepends=True)
        if lines:
            header = lines[0]
            existing = lines[1:]

    keep: list[str] = []
    replaced = False
    month_prefix = month + ","
    for r in existing:
        if r.startswith(month_prefix):
            try:
                prev_total = float(r.split(",")[_DRIFT_TOTAL_COL])
            except (IndexError, ValueError):
                prev_total = -1.0
            if sr_total <= prev_total:
                # Existing row is at least as good — leave the file untouched.
                return
            replaced = True
            continue  # drop the old row; the new one is appended below
        keep.append(r)
    keep.append(row)

    with _DRIFT_FILE.open("w") as f:
        f.write(header)
        f.writelines(keep)

    tag = "overwrote" if replaced else "appended"
    print(f"  Drift snapshot {tag} for {month} "
          f"(IT income success {sr_total * 100:.1f}%) → {_DRIFT_FILE}")


def _plot_drift_history() -> None:
    """Render drift_history.png from drift_history.csv: success-rate-it-total vs
    time, with the latest snapshot's plan parameters in the title. Lazy-imports
    matplotlib so it's only loaded when this runs; skips silently if the file
    has only a header or matplotlib isn't installed."""
    if not _DRIFT_FILE.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless write-only
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [!] Skipping drift plot — `pip install matplotlib` to enable.")
        return
    from datetime import datetime

    lines = _DRIFT_FILE.read_text().splitlines()
    if len(lines) < 2:
        return
    rows: list[tuple[datetime, float, int, float, float]] = []
    for raw in lines[1:]:
        parts = raw.split(",")
        if len(parts) <= _DRIFT_TOTAL_COL:
            continue
        try:
            dt = datetime.strptime(parts[0], "%m-%Y")
            horizon = int(parts[2])
            target_income = float(parts[3])
            target_portfolio = float(parts[4])
            total = float(parts[_DRIFT_TOTAL_COL])
        except (ValueError, IndexError):
            continue
        rows.append((dt, total, horizon, target_income, target_portfolio))
    if not rows:
        return
    rows.sort(key=lambda r: r[0])
    dates = [r[0] for r in rows]
    success_pct = [r[1] * 100 for r in rows]

    # Title shows the latest snapshot's plan parameters — that's the current intent.
    _, _, horizon, target_income, target_portfolio = rows[-1]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(dates, success_pct, marker="o", linewidth=1.6, color="#1f77b4")
    ax.set_ylim(0, 100)
    ax.set_ylabel("IT income success rate (%)")
    ax.set_xlabel("Month")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    ax.set_title(
        f"Portfolio drift — IT income success rate over time\n"
        f"horizon {horizon}y · target €{target_income:,.0f}/mo · "
        f"target portfolio €{target_portfolio:,.0f}",
        fontsize=10,
    )
    out = _DRIFT_FILE.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Drift plot saved → {out}")


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
    realism: CagrRealismSnapshot | None = None,
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
        "realism":       asdict(realism)       if realism       else None,
    }
    _SUMMARIES_FILE.write_text(json.dumps(payload, indent=2))


def _load_context() -> tuple[
    str, str, str, float | None,
    DecumulationSummary | None, PlanVerdictSummary | None, CagrRealismSnapshot | None,
]:
    if not _CONTEXT_FILE.exists():
        return "", "", "", None, None, None, None
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
    realism:       CagrRealismSnapshot | None = None
    if _SUMMARIES_FILE.exists():
        import json
        payload = json.loads(_SUMMARIES_FILE.read_text())
        if payload.get("decum_summary"):
            decum_summary = DecumulationSummary(**payload["decum_summary"])
        if payload.get("plan_verdict"):
            plan_verdict = PlanVerdictSummary(**payload["plan_verdict"])
        if payload.get("realism"):
            realism = CagrRealismSnapshot(**payload["realism"])
    return accum, decum, cagr, required_cagr, decum_summary, plan_verdict, realism


# ── Sections ──────────────────────────────────────────────────────────────────

def _planning_cagr_argv(planning_cagr: str | None) -> list[str]:
    """Return ['--cagr', X] if set, else []. Used to propagate to sub-scripts."""
    return ["--cagr", planning_cagr] if planning_cagr else []


def prep_canonical_fi(planning_cagr: str | None = None) -> CanonicalFIRequiredScan | None:
    """Compute the canonical (preservation-calibrated) FI bar for decumulation's
    anchor display. Accumulation no longer targets this number — it sizes against
    the income-formula required portfolio (W₀ = target_income / yield) so the
    accum takeaway stays actionable. The preservation bar is shown as reference
    in decumulation only. Returns None if config is unavailable."""
    import tomllib
    from models import Config
    from monte_carlo import compute_canonical_fi_required_scan

    try:
        with open("config.toml", "rb") as f:
            raw = tomllib.load(f)
    except Exception as e:
        print(f"  [!] Skipping canonical FI prep — couldn't load config.toml: {e}")
        return None
    config = Config.from_dict(raw)
    config.planning_cagr_override = Config.parse_planning_cagr(planning_cagr)

    infl_factor = (1.0 + config.expected_inflation_percent / 100.0) ** config.years
    target_monthly_nom = config.target_monthly_retirement_income * infl_factor

    print()
    print("─" * 80)
    print("FI CANONICAL BAR  —  preservation-calibrated required portfolio")
    print("─" * 80)
    print(f"  Preserve ≥{int(round(config.fi_preservation_ratio * 100))}% of initial wealth "
          f"at yr {config.mc_decumulation_years}")
    print(f"  Baseline P≥{int(round(config.fi_preservation_confidence_baseline * 100))}%, "
          f"worst-SORR P≥{int(round(config.fi_preservation_confidence_stress * 100))}%")
    print(f"  Computing scan…", end="", flush=True)
    scan = compute_canonical_fi_required_scan(config, target_monthly_nom)
    label = "reachable" if scan.canonical_reachable else "UNREACHABLE in search range"
    print(f"\r  Result: €{scan.canonical_required_wealth:,.0f}  "
          f"at {int(round(scan.canonical_optimal_bonds_pct * 100))}% bonds  "
          f"(worst-SORR: {scan.worst_stress_label})  [{label}]                      ")
    print("  → decumulation will use this as its anchor for the focused 3-question block.")
    print("  Note: this dual-confidence bar adds a worst-SORR floor on top of the baseline")
    print("        preservation confidence. Decumulation's body reports a separate focused-Q3")
    print("        single-confidence number (typically smaller) — both are correct under their")
    print("        own definitions; the gap is the cost of the SORR safety margin.")
    print()
    return scan


def run_accum(verbose: bool = False, planning_cagr: str | None = None,
              ):
    """Run accumulation section.

    Returns (report_text, p50, tail, required_cagr, tail_pct, accum_required_it, verdict_inputs).
    accum_required_it is the IT scenario's income-formula required portfolio
    (€2.07M at config) — passed to decum's chained MC for the §4.8 secondary threshold.
    verdict_inputs is None on failure; otherwise a VerdictInputs bundle for print_final_verdict.
    """
    import main_portfolio_projection as m

    _banner(1, "ACCUMULATION — target portfolio, contributions, financial freedom")
    argv = ["--no-clear", *_planning_cagr_argv(planning_cagr)]
    if verbose:
        argv.append("--verbose")
    text, result = _capture(m.main, argv)

    if result is None:
        print("\n  [!] Accumulation failed — check config.toml")
        return text, 0.0, 0.0, None, 3, 0.0, None
    p50, tail, required_cagr, tail_pct, accum_required_it, _vr, _cs, _cfg, verdict_inputs = result
    return text, p50, tail, required_cagr, tail_pct, accum_required_it, verdict_inputs


def run_decum(p50: float,
              accum_required_it: float = 0.0,
              canonical_scan: CanonicalFIRequiredScan | None = None,
              planning_cagr: str | None = None) -> tuple[str, DecumulationSummary | None]:
    """Run decumulation section. Returns (report text, typed summary).

    p50: accumulation P50 wealth from main_portfolio_projection — used as the
         starting wealth for survival curves, fan, GK range, SORR, and §4.11.
    accum_required_it: IT scenario's income-formula required portfolio — used as
         the §4.8 secondary 'reaching ≥' threshold for cross-comparison with accum's
         success rate.
    canonical_scan: precomputed canonical FI scan; if provided, decumulation skips
         the internal scan and uses this one as the anchor.
    """
    import main_decumulation as m

    print()
    _banner(2, "DECUMULATION — withdrawal survival & wealth preservation under SORR")
    text, summary = _capture(
        lambda: m.main(
            portfolio_override=p50 if p50 > 0 else None,
            accum_required_override=accum_required_it if accum_required_it > 0 else None,
            canonical_fi_scan=canonical_scan,
            no_clear=True,
            cagr=planning_cagr,
        ),
    )
    return text, summary if isinstance(summary, DecumulationSummary) else None


def run_cagr(planning_cagr: str | None = None
             ) -> tuple[str, object, CagrRealismSnapshot | None]:
    """Run CAGR / instrument section.

    Returns (agent_summary_text, OptimalPortfolioResult | None, CagrRealismSnapshot | None).
    The text is trimmed to the AGENT SUMMARY section to keep agent context compact.
    """
    import main_cagr_analysis as m

    _banner(3, "INSTRUMENTS — historical CAGR, volatility, correlation + optimiser")
    text, result = _capture(m.main, ["--no-clear", *_planning_cagr_argv(planning_cagr)])

    # result is (list[TickerResult], OptimalPortfolioResult | None, CagrRealismSnapshot | None)
    opt_result = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    realism    = result[2] if isinstance(result, tuple) and len(result) > 2 else None
    if not isinstance(realism, CagrRealismSnapshot):
        realism = None

    marker = "AGENT SUMMARY"
    agent_text = text[text.index(marker):] if marker in text else text
    return agent_text, opt_result, realism


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
    parser.add_argument("--cagr", default=None,
                        help="override stock CAGR for all sections "
                             "(accepts 4, 4.0, 0.04, '4%%'); propagated to sub-scripts")
    args = parser.parse_args()

    from reports.formatters import clear_terminal
    clear_terminal()

    start = _SECTIONS.index(args.start_from)
    planning_cagr = args.cagr  # string or None — propagated verbatim

    # Load cached context for sections we're skipping
    accum_text, decum_text, cagr_text, required_cagr, decum_summary, plan_verdict, realism = _load_context()
    p50 = tail = 0.0
    tail_pct = 3
    config_toml = Path("config.toml").read_text() if Path("config.toml").exists() else ""

    # Preflight: canonical FI bar — both pillars target the same preservation-calibrated number.
    canonical_scan: CanonicalFIRequiredScan | None = None
    if start <= _SECTIONS.index("decum"):
        canonical_scan = prep_canonical_fi(planning_cagr=planning_cagr)

    # Section 1: Accumulation
    accum_required_it = 0.0
    verdict_inputs = None
    if start <= _SECTIONS.index("accum"):
        accum_text, p50, tail, required_cagr, tail_pct, accum_required_it, verdict_inputs = run_accum(
            verbose=args.verbose, planning_cagr=planning_cagr,
        )

    # Section 2: Decumulation
    if start <= _SECTIONS.index("decum"):
        decum_text, decum_summary = run_decum(p50,
                                              accum_required_it=accum_required_it,
                                              canonical_scan=canonical_scan,
                                              planning_cagr=planning_cagr)

    # Section 3: CAGR + optimiser
    if start <= _SECTIONS.index("cagr"):
        cagr_text, _opt, realism = run_cagr(planning_cagr=planning_cagr)

    # Cache context for --from agent re-runs
    _save_context(accum_text, decum_text, cagr_text, required_cagr, decum_summary, plan_verdict, realism)

    # Final consolidated verdict — pulls from accum + decum + cagr-realism, no new MC.
    if verdict_inputs is not None:
        from reports.summary import print_final_verdict
        print_final_verdict(verdict_inputs, decum_summary, realism)

    # Drift log — one row per full pipeline run
    joint_P = decum_summary.joint_survival_chained if decum_summary is not None else None
    _append_history(required_cagr=required_cagr, joint_P=joint_P, p50=p50)

    # Monthly drift snapshot — keyed by MM-YYYY, kept only when success improves.
    if verdict_inputs is not None:
        _append_drift_snapshot(verdict_inputs, accum_required_it)
        _plot_drift_history()

    # Section 4: AI Agent — only when explicitly requested
    if args.agent or args.start_from == "agent":
        run_agent(accum_text, decum_text, cagr_text, required_cagr, config_toml)


if __name__ == "__main__":
    main()
