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
import io
import os
import shutil
import tempfile
import tomllib
from pathlib import Path

from main import run_accum
from models import Config, resolve_config_path
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

    # Resolved values actually used for this call - the saved-plan value for
    # anything the caller omitted, the override for anything it passed. Read
    # independently of run_accum() below so it's still available to explain
    # *why* a run failed (e.g. a contradictory saved-plan value), not just
    # to report a successful one.
    with open(resolve_config_path(), "rb") as f:
        base_config = tomllib.load(f)
    assumptions_used = {k: base_config[k] for k in _ACCUM_WHITELIST}
    assumptions_used.update(overrides)
    resolved_cagr = Config.parse_planning_cagr(cagr)
    assumptions_used["stock_cagr_percent"] = (
        resolved_cagr * 100 if resolved_cagr is not None
        else base_config["expected_yearly_growth_stocks_percent"]
    )
    assumptions_used["overrides_applied"] = overrides  # what the caller actually changed

    prev_fire_config = os.environ.get("FIRE_CONFIG")
    tmp_dir_ctx = tempfile.TemporaryDirectory(prefix="fire_mcp_") if overrides else None
    try:
        if overrides:
            tmp_config = Path(tmp_dir_ctx.name) / "config.toml"
            shutil.copy(Path(resolve_config_path()), tmp_config)
            error = apply_cli_overrides(tmp_config, overrides)
            if error:
                raise ValueError(error)
            # os.environ is process-global, so this assumes calls into this
            # tool don't overlap - true today (stdio is one request at a time
            # from a single client) but not guaranteed if a second tool that
            # touches FIRE_CONFIG is added, or the transport changes. Add a
            # lock around the override→call→restore sequence before that.
            os.environ["FIRE_CONFIG"] = str(tmp_config)

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
    finally:
        if overrides:
            if prev_fire_config is None:
                os.environ.pop("FIRE_CONFIG", None)
            else:
                os.environ["FIRE_CONFIG"] = prev_fire_config
            tmp_dir_ctx.cleanup()

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


@server.prompt()
def projection() -> str:
    """Portfolio lifecycle projection only: run the portfolio lifecycle simulation and return a summary."""
    return (
        "Give me a portfolio-lifecycle-analysis review: current situation, plan and risks."
    )


if __name__ == "__main__":
    asyncio.run(server.run_stdio_async())
