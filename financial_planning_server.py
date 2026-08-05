"""Financial planning MCP server: drives the retirement Monte Carlo engine over
stdio for an LLM agent (see ../AIagent, which spawns this as one of its MCP
servers). Read-only — every tool here only computes and returns a result,
never writes to config.toml or any of the run-history / drift-tracking files.

Structure - no class, no context manager, unlike the client side (mcp_client.py
in AIagent):
  - `server` (below) is one module-level MCPServer instance.
  - Each function decorated `@server.tool()` or `@server.prompt()` becomes
    visible to a connected client's list_tools()/list_prompts(); everything
    else is a private helper, invisible to the client.
  - `if __name__ == "__main__":` calls `asyncio.run(server.run_stdio_async())`,
    which blocks for the process's entire life: read a request from stdin,
    dispatch it to the matching decorated function, write the response to
    stdout, repeat. There's no __aenter__/__aexit__ pair here because the
    server doesn't initiate a connection - it just runs until stdin closes,
    which happens when the client's __aexit__ closes its end of the pipe.
"""
import asyncio
import contextlib
import dataclasses
import io
import os
import shutil
import tempfile
import tomllib
from pathlib import Path

from main import run_accum, run_decum
from models import Config, DecumulationSummary, resolve_config_path
from settings_gui import apply_cli_overrides

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("portfolio-lifecycle-simulator")

_ACCUM_WHITELIST = {
    "target_monthly_retirement_income", "years", "bonds_percentage",
    "annual_net_saving", "current_stock_value", "current_bonds_value",
    "current_cash", "withdrawal_percentage_stock",
    "stock_monthly_investment_pct", "bonds_monthly_investment_pct",
}

_DECUM_WHITELIST = {
    "target_monthly_retirement_income", "years", "bonds_percentage",
    "withdrawal_percentage_stock", "mc_decumulation_years",
    "fi_preservation_ratio", "fi_preservation_confidence_baseline",
    "fi_preservation_confidence_stress",
}


def _resolve_assumptions(whitelist: set[str], overrides: dict[str, float], cagr: str | None) -> dict:
    """The full set of plan-lever values actually used for this call: the
    saved-plan value (config.toml, or whatever FIRE_CONFIG points at) for
    anything the caller omitted, the override for anything it passed. Read
    independently of the tool's own compute call so it's available to
    explain *why* a run failed (e.g. a contradictory saved-plan value), not
    just to annotate a successful one."""
    with open(resolve_config_path(), "rb") as f:
        base_config = tomllib.load(f)
    assumptions_used = {k: base_config[k] for k in whitelist}
    assumptions_used.update(overrides)
    resolved_cagr = Config.parse_planning_cagr(cagr)
    assumptions_used["stock_cagr_percent"] = (
        resolved_cagr * 100 if resolved_cagr is not None
        else base_config["expected_yearly_growth_stocks_percent"]
    )
    assumptions_used["overrides_applied"] = overrides  # what the caller actually changed
    return assumptions_used


@contextlib.contextmanager
def _isolated_config_override(overrides: dict[str, float]):
    """While inside the `with` block, FIRE_CONFIG points at a temp copy of
    config.toml with `overrides` merged in (same validation the GUI dialog
    uses); restores the previous FIRE_CONFIG state on exit either way. No-op
    (real config.toml untouched, FIRE_CONFIG untouched) if overrides is empty.

    os.environ is process-global, so this assumes calls into tools that use
    it don't overlap - true today (stdio is one request at a time from a
    single client) but not guaranteed if the transport ever changes. Add a
    lock around this if that stops being true.
    """
    if not overrides:
        yield
        return
    prev_fire_config = os.environ.get("FIRE_CONFIG")
    with tempfile.TemporaryDirectory(prefix="fire_mcp_") as tmp_dir:
        tmp_config = Path(tmp_dir) / "config.toml"
        shutil.copy(Path(resolve_config_path()), tmp_config)
        error = apply_cli_overrides(tmp_config, overrides)
        if error:
            raise ValueError(error)
        os.environ["FIRE_CONFIG"] = str(tmp_config)
        try:
            yield
        finally:
            if prev_fire_config is None:
                os.environ.pop("FIRE_CONFIG", None)
            else:
                os.environ["FIRE_CONFIG"] = prev_fire_config

_BASE_DESCRIPTION = """Answers the core planning question: what portfolio is required at the
horizon to cover a target monthly income, and is the current plan on
track to reach it?

Returns five things:
 1. Required portfolio at the horizon, for German and Italian tax
    residence. The user plans to retire in Italy — quote IT unless
    asked otherwise.
 2. Current success probability, for portfolio and for income.
 3. Minimum contributions or cash redeployment that would raise it.
 4. The same, for the income target specifically.
 5. The two assumptions the outcome is most sensitive to.

EVERY PARAMETER IS OPTIONAL. Omitted parameters use the user's saved
plan — portfolio, savings rate, withdrawal rate, inflation, horizon,
allocation. Do not ask the user for these; pass only the values they
explicitly changed in this conversation."""


def _assumption_summary() -> str:
    """Live plan values, injected into the tool description at server start.
    Evaluated once, when the decorator below runs - config.toml doesn't
    change during a server's lifetime, so this is a one-time snapshot, not
    something re-read per call."""
    try:
        with open(resolve_config_path(), "rb") as f:
            cfg = Config.from_dict(tomllib.load(f))
        return (
            "\n\nCURRENT SAVED PLAN (used for any omitted parameter): "
            f"horizon {cfg.years:.0f}y, "
            f"target income €{cfg.target_monthly_retirement_income:,.0f}/mo (today's money), "
            f"withdrawal rate {cfg.withdrawal_percentage_stock}%/yr, "
            f"{cfg.bonds_percentage:.0f}% bonds, "
            f"inflation {cfg.expected_inflation_percent}%, "
            f"stock CAGR {cfg.expected_yearly_growth_stocks_percent}%."
        )
    except Exception:                      # importable without config.toml
        return "\n\nOmitted parameters use the user's saved plan."


@server.tool(description=_BASE_DESCRIPTION + _assumption_summary(),
             annotations=ToolAnnotations(readOnlyHint=True))
def accumulation_projection(
    target_monthly_retirement_income: float | None = None,
    years: float | None = None,
    bonds_percentage: float | None = None,
    annual_net_saving: float | None = None,
    current_stock_value: float | None = None,
    current_bonds_value: float | None = None,
    current_cash: float | None = None,
    withdrawal_percentage_stock: float | None = None,
    stock_monthly_investment_pct: float | None = None,
    bonds_monthly_investment_pct: float | None = None,
    cagr: str | None = None,
    include_report: bool = False,
) -> dict:
    """Protocol-visible description comes from _BASE_DESCRIPTION + the live
    plan snapshot (see the @server.tool decorator above), not this docstring -
    kept short here to avoid two copies of the same text drifting apart."""
    # locals() here only contains this function's parameters because this is
    # the first statement in the body - if a local variable is ever added
    # above this line, it becomes eligible for `overrides` too. Keep this
    # first, or switch to an explicit dict.
    overrides = {k: v for k, v in locals().items()
                 if k in _ACCUM_WHITELIST and v is not None}
    assumptions_used = _resolve_assumptions(_ACCUM_WHITELIST, overrides, cagr)

    with _isolated_config_override(overrides):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # verbose=False here is not this tool's include_report - it picks
            # which report main_portfolio_projection renders: False -> the
            # concise "TAKE AWAY" text (what "report" below is), True -> a much
            # larger "clean_summary" (grids, heatmaps). We always want the
            # former, so this is pinned, not forwarded from a caller flag.
            text, p50, tail, required_cagr, tail_pct, accum_required_it, _verdict_inputs = (
                run_accum(verbose=False, planning_cagr=cagr)
            )

    result = {
        "assumptions_used": assumptions_used,
        "p50_portfolio_eur": p50,
        "tail_portfolio_eur": tail,
        "tail_percentile": tail_pct,
        "required_stock_cagr": required_cagr,
        "required_portfolio_income_bar_eur": accum_required_it,
    }
    if include_report:
        result["report"] = text
    return result


_DECUM_BASE_DESCRIPTION = """Does the portfolio survive and preserve its real value through the
withdrawal phase? Answers the three focused decumulation questions,
anchored on the IT required portfolio from the income-formula bar:

 1. At the IT required portfolio, what's the probability the real final
    wealth still meets the preservation bar (preserve_ratio · W)?
 2. What monthly contribution would grow the deterministic projection up
    to the preservation-required wealth?
 3. What is the preservation-required wealth at the best allocation?

Also returns the years-to-retirement tradeoff (max safe withdrawal at
each candidate retirement year) and a preservation-ratio sensitivity scan.

EVERY PARAMETER IS OPTIONAL. Omitted parameters use the user's saved
plan. Do not ask the user for these; pass only the values they explicitly
changed in this conversation."""


def _decum_assumption_summary() -> str:
    """Live plan values, injected into the tool description at server start.
    Same one-time-snapshot reasoning as _assumption_summary() above."""
    try:
        with open(resolve_config_path(), "rb") as f:
            cfg = Config.from_dict(tomllib.load(f))
        return (
            "\n\nCURRENT SAVED PLAN (used for any omitted parameter): "
            f"decumulation horizon {cfg.mc_decumulation_years}y, "
            f"preservation ratio {cfg.fi_preservation_ratio:.0%}, "
            f"confidence baseline {cfg.fi_preservation_confidence_baseline:.0%} "
            f"/ worst-SORR {cfg.fi_preservation_confidence_stress:.0%}, "
            f"{cfg.bonds_percentage:.0f}% bonds."
        )
    except Exception:                      # importable without config.toml
        return "\n\nOmitted parameters use the user's saved plan."


@server.tool(description=_DECUM_BASE_DESCRIPTION + _decum_assumption_summary(),
             annotations=ToolAnnotations(readOnlyHint=True))
def decumulation_projection(
    target_monthly_retirement_income: float | None = None,
    years: float | None = None,
    bonds_percentage: float | None = None,
    withdrawal_percentage_stock: float | None = None,
    mc_decumulation_years: float | None = None,
    fi_preservation_ratio: float | None = None,
    fi_preservation_confidence_baseline: float | None = None,
    fi_preservation_confidence_stress: float | None = None,
    cagr: str | None = None,
    include_report: bool = False,
) -> dict:
    """Protocol-visible description comes from _DECUM_BASE_DESCRIPTION + the
    live plan snapshot (see the @server.tool decorator above), not this
    docstring - kept short here to avoid two copies of the same text
    drifting apart."""
    # locals() here only contains this function's parameters because this is
    # the first statement in the body - see the identical note on
    # accumulation_projection above.
    overrides = {k: v for k, v in locals().items()
                 if k in _DECUM_WHITELIST and v is not None}
    assumptions_used = _resolve_assumptions(_DECUM_WHITELIST, overrides, cagr)

    with _isolated_config_override(overrides):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # verbose=False -> the focused Takeaway + survival matrix (§4.14).
            # verbose=True would additionally print the Q1/Q2/Q3 tables and
            # §4.10-§4.13/§4.15 supporting context - pinned False for the same
            # "return structured numbers, not a caller-controlled report
            # size" reasoning as accumulation_projection.
            # p50=0.0 -> portfolio_override=None inside run_decum(): the
            # focused pipeline self-anchors on the IT required portfolio, it
            # doesn't need an external starting-wealth figure when not verbose
            # (portfolio_override is only read inside main_decumulation's
            # verbose-only §4.10/§4.11 supporting-context block).
            text, summary = run_decum(p50=0.0, planning_cagr=cagr, verbose=False)

    result = {"assumptions_used": assumptions_used}
    if summary is not None:
        result.update(dataclasses.asdict(summary))
    else:
        result.update({f.name: None for f in dataclasses.fields(DecumulationSummary)})
    if include_report:
        result["report"] = text
    return result


@server.prompt()
def projection() -> str:
    """Portfolio lifecycle projection only: run the portfolio lifecycle simulation and return a summary."""
    return (
        "Give me a portfolio-lifecycle-analysis review: current situation, plan and risks."
    )


if __name__ == "__main__":
    asyncio.run(server.run_stdio_async())
