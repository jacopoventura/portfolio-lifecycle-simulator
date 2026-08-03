"""
main.py — unified retirement planning analysis

Computes the full pipeline silently (live progress markers only), then prints a
verdict-first report: a self-contained RETIREMENT PLAN CHECK block (3 pillars,
since-last-run delta, levers, three-bars legend) followed by the evidence
sections:

  S1. ACCUMULATION  — do I reach the income bar by the plan date?
  S2. DECUMULATION  — does that wealth survive & preserve real value?
  S3. INSTRUMENTS   — does the real world deliver the assumed CAGR?
  S4. AI AGENT      — action plan integrating all sections (opt-in via --agent)

Usage:
  python main.py                     # parameter dialog, then sections 1→2→3 (no AI call)
  python main.py --no-gui            # skip the parameter dialog, use config.toml as-is
  python main.py --agent             # full pipeline including AI agent (section 4)
  python main.py --from decum        # start from decumulation (uses config values for p50/tail)
  python main.py --from cagr         # CAGR only
  python main.py --from agent        # re-run agent on last captured context
  python main.py --verbose           # pass --verbose to accumulation and decumulation sections

Non-interactive / AI-agent use: every value editable in the GUI dialog is also
a CLI flag (see `--help`, "plan parameters" group) — passing any of them skips
the dialog entirely, so an agent can drive the full pipeline without a display:
  python main.py --no-gui --target-monthly-retirement-income 3000 --years 16 \\
      --bonds-percentage 50 --current-stock-value 108000 ...

Requires ANTHROPIC_API_KEY in .env for section 4.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

from models import CagrRealismSnapshot, DecumulationSummary, PlanVerdictSummary, resolve_config_path


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
) -> str:
    """Save a per-month snapshot of plan state + simulation outcome.

    One row per calendar month (MM-YYYY). When the same month already has a
    row, we overwrite it ONLY if the new IT income success rate beats what's
    on file — so the log becomes "best snapshot so far this month" rather
    than a noisy per-run dump. Returns a status string for the report footer."""
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
                return (f"snapshot kept — {prev_total * 100:.1f}% on file "
                        f"beats {sr_total * 100:.1f}%")
            replaced = True
            continue  # drop the old row; the new one is appended below
        keep.append(r)
    keep.append(row)

    with _DRIFT_FILE.open("w") as f:
        f.write(header)
        f.writelines(keep)

    tag = "overwrote" if replaced else "appended"
    return f"snapshot {tag} for {month} (income success {sr_total * 100:.1f}%)"


def _plot_drift_history() -> str | None:
    """Render drift_history.png from drift_history.csv: success-rate-it-total vs
    time, with the latest snapshot's plan parameters in the title. Lazy-imports
    matplotlib so it's only loaded when this runs; skips silently if the file
    has only a header or matplotlib isn't installed. Returns the written
    filename for the report footer, or None when skipped."""
    if not _DRIFT_FILE.exists():
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless write-only
        import matplotlib.pyplot as plt
    except ImportError:
        return "drift plot skipped — pip install matplotlib"
    from datetime import datetime

    lines = _DRIFT_FILE.read_text().splitlines()
    if len(lines) < 2:
        return None
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
        return None
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
    return out.name


def _read_prev_drift_row() -> dict | None:
    """Most recent drift row from a month BEFORE the current one — feeds the
    'since last run' delta in the plan-check block. None when no history."""
    if not _DRIFT_FILE.exists():
        return None
    from datetime import date, datetime
    current_month = date.today().strftime("%m-%Y")
    best: dict | None = None
    best_dt: "datetime | None" = None
    for raw in _DRIFT_FILE.read_text().splitlines()[1:]:
        parts = raw.split(",")
        if len(parts) <= _DRIFT_TOTAL_COL or parts[0] == current_month:
            continue
        try:
            dt = datetime.strptime(parts[0], "%m-%Y")
            row = {
                "month": parts[0],
                "bonds_pct": float(parts[1]),
                "horizon": int(parts[2]),
                "target_income": float(parts[3]),
                "sr_total": float(parts[_DRIFT_TOTAL_COL]),
            }
        except (ValueError, IndexError):
            continue
        if best_dt is None or dt > best_dt:
            best_dt, best = dt, row
    return best


_SECTIONS = ("accum", "decum", "cagr", "agent")


def _banner(n: int, title: str) -> None:
    print()
    print("─" * 80)
    print(f"SECTION {n}  —  {title}")
    print("─" * 80)


def _evidence_banner(tag: str, question: str, feeds: str) -> None:
    """Section header for the evidence half of the report: the question the
    section answers plus which verdict pillar / lever it feeds."""
    print()
    print("─" * 80)
    print(f"{tag} — {question}")
    print(f"     feeds: {feeds}")
    print("─" * 80)


def _capture(fn, argv_extra: list[str] | None = None):
    """Call fn() with a clean sys.argv, capturing stdout silently.
    Returns (text, return_value). Output is replayed later in the
    verdict-first report — nothing prints live except progress markers."""
    buf = io.StringIO()
    saved = sys.argv
    sys.argv = [saved[0]] + (argv_extra or [])
    try:
        with contextlib.redirect_stdout(buf):
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


def run_accum(verbose: bool = False, planning_cagr: str | None = None,
              ):
    """Run accumulation section.

    Returns (report_text, p50, tail, required_cagr, tail_pct, accum_required_it, verdict_inputs).
    accum_required_it is the IT scenario's income-formula required portfolio
    (€2.07M at config) — passed to decum's chained MC for the §4.8 secondary threshold.
    verdict_inputs is None on failure; otherwise a VerdictInputs bundle for print_final_verdict.
    """
    import main_portfolio_projection as m

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
              planning_cagr: str | None = None,
              verbose: bool = False) -> tuple[str, DecumulationSummary | None]:
    """Run decumulation section. Returns (report text, typed summary).

    p50: accumulation P50 wealth from main_portfolio_projection — used as the
         starting wealth for survival curves, fan, GK range, SORR, and §4.11.
    verbose: if True, also prints §4.10-§4.15 (inflation×CAGR grid, SORR
        preservation matrix, price-of-constraint, two-bucket, survival matrix,
        crash timing) — otherwise only the Takeaway shows.
    """
    import main_decumulation as m

    text, summary = _capture(
        lambda: m.main(
            portfolio_override=p50 if p50 > 0 else None,
            no_clear=True,
            cagr=planning_cagr,
            verbose=verbose,
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
    from settings_gui import GUI_FIELDS, apply_cli_overrides

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
                        help="pass --verbose to accumulation and decumulation sections")
    parser.add_argument("--cagr", default=None,
                        help="override stock CAGR for all sections "
                             "(accepts 4, 4.0, 0.04, '4%%'); propagated to sub-scripts")
    parser.add_argument("--no-gui", action="store_true",
                        help="skip the parameter dialog and run with config.toml as-is "
                             "(implied automatically if any plan-parameter flag below is passed)")

    plan_params = parser.add_argument_group(
        "plan parameters",
        "Every value editable in the GUI dialog, exposed as a flag for non-interactive "
        "/ AI-agent use. Passing ANY of these skips the dialog entirely — unspecified "
        "values keep whatever is already in config.toml.",
    )
    for _key, _label, _unit in GUI_FIELDS:
        plan_params.add_argument(
            "--" + _key.replace("_", "-"), type=float, default=None,
            help=f"{_label} ({_unit})".replace("%", "%%"),
        )

    args = parser.parse_args()

    param_overrides = {
        key: getattr(args, key)
        for key, _label, _unit in GUI_FIELDS
        if getattr(args, key) is not None
    }

    if param_overrides:
        # Isolate exploratory / AI-agent overrides to a temp copy — never the
        # real config.toml. Twenty "what if I retire at 60" tool calls should
        # leave zero permanent mutations of the user's actual plan.
        import shutil
        import tempfile

        real_config = Path(resolve_config_path())
        tmp_dir = Path(tempfile.mkdtemp(prefix="fire_config_"))
        tmp_config = tmp_dir / "config.toml"
        shutil.copy(real_config, tmp_config)
        error = apply_cli_overrides(tmp_config, param_overrides)
        if error:
            print(f"Invalid parameter: {error}")
            return
        os.environ["FIRE_CONFIG"] = str(tmp_config)
    elif not args.no_gui:
        try:
            from settings_gui import edit_settings
            if not edit_settings(Path(resolve_config_path())):
                print("Cancelled — no simulation run.")
                return
        except Exception as exc:  # headless / tkinter unavailable
            print(f"(parameter dialog unavailable: {exc} — running with config.toml as-is)")

    from reports.formatters import clear_terminal, format_millions
    clear_terminal()

    start = _SECTIONS.index(args.start_from)
    planning_cagr = args.cagr  # string or None — propagated verbatim

    # Load cached context for sections we're skipping
    accum_text, decum_text, cagr_text, required_cagr, decum_summary, plan_verdict, realism = _load_context()
    p50 = tail = 0.0
    tail_pct = 3
    _config_path = Path(resolve_config_path())
    config_toml = _config_path.read_text() if _config_path.exists() else ""

    # ── Compute all sections silently, with live progress markers ────────────
    print("RETIREMENT PLAN CHECK — computing… (report prints when all sections finish)")

    accum_required_it = 0.0
    verdict_inputs = None
    if start <= _SECTIONS.index("accum"):
        print("  [1/3] accumulation…", flush=True)
        accum_text, p50, tail, required_cagr, tail_pct, accum_required_it, verdict_inputs = run_accum(
            verbose=args.verbose, planning_cagr=planning_cagr,
        )

    if start <= _SECTIONS.index("decum"):
        print("  [2/3] decumulation…", flush=True)
        decum_text, decum_summary = run_decum(p50, planning_cagr=planning_cagr, verbose=args.verbose)

    if start <= _SECTIONS.index("cagr"):
        print("  [3/3] instruments…", flush=True)
        cagr_text, _opt, realism = run_cagr(planning_cagr=planning_cagr)

    # Cache context for --from agent re-runs — skipped for exploratory CLI-override
    # runs (param_overrides), so "what if" scenarios never overwrite the real
    # cached context, run history, or drift tracking.
    if not param_overrides:
        _save_context(accum_text, decum_text, cagr_text, required_cagr, decum_summary, plan_verdict, realism)

    # ── Verdict first — 30-second top block, fully self-contained ────────────
    print()
    if verdict_inputs is not None:
        from reports.summary import print_plan_check
        print_plan_check(
            verdict_inputs, decum_summary, realism,
            income_bar=accum_required_it,
            drift_prev=_read_prev_drift_row(),
        )
    else:
        print("  (accumulation skipped — no verdict block; evidence sections below "
              "use cached or partial data)")

    # ── Evidence — the detailed sections behind the verdict ──────────────────
    bar = "═" * 80
    print(bar)
    print("  EVIDENCE — details behind the verdict")
    print(bar)

    from datetime import date
    cfg = verdict_inputs.config if verdict_inputs is not None else None
    retire_year = date.today().year + int(round(cfg.years)) if cfg is not None else None
    income_bar_txt = (f"the income min ({format_millions(accum_required_it)})"
                      if accum_required_it > 0 else "the income min")

    _evidence_banner(
        "S1 · ACCUMULATION",
        f"do I reach {income_bar_txt} by {retire_year or 'retirement'}?",
        "INCOME pillar · lever A (save more) · lever C (redeploy)",
    )
    print(accum_text, end="")

    end_year = (retire_year + cfg.mc_decumulation_years
                if cfg is not None and retire_year is not None else None)
    _evidence_banner(
        "S2 · DECUMULATION",
        f"does {retire_year or 'retirement'} wealth survive & preserve real value "
        f"to {end_year or 'end of plan'}?",
        "PRESERVATION pillar · lever B (retire later)",
    )
    print(decum_text, end="")

    cagr_q = (f"does the real world deliver the {cfg.planning_cagr * 100:.1f}% I assumed?"
              if cfg is not None else "does the real world deliver the assumed CAGR?")
    _evidence_banner(
        "S3 · INSTRUMENTS", cagr_q,
        "RETURN pillar · ETF picks for the stock sleeve",
    )
    print(cagr_text, end="")

    # ── History + drift + footer ──────────────────────────────────────────────
    # Exploratory CLI-override runs never touch run_history.csv or
    # drift_history.csv either — same "no permanent mutation from a what-if"
    # guarantee as the config isolation above.
    file_notes: list[str] = []
    if not param_overrides:
        joint_P = decum_summary.joint_survival_chained if decum_summary is not None else None
        _append_history(required_cagr=required_cagr, joint_P=joint_P, p50=p50)

        if verdict_inputs is not None:
            drift_status = _append_drift_snapshot(verdict_inputs, accum_required_it)
            file_notes.append(f"drift_history.csv ({drift_status})")
            plot_note = _plot_drift_history()
            if plot_note:
                file_notes.append(plot_note)
        file_notes.append("last_run_context.txt")
    else:
        file_notes.append("none (exploratory run — config/history untouched)")

    print()
    print("─" * 80)
    print(f"  Files: {' · '.join(file_notes)}")
    print("─" * 80)

    # Section 4: AI Agent — only when explicitly requested
    if args.agent or args.start_from == "agent":
        run_agent(accum_text, decum_text, cagr_text, required_cagr, config_toml)


if __name__ == "__main__":
    main()
