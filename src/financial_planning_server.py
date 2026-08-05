"""Sheets MCP server: read-only access to the portfolio spreadsheet over stdio.

Auth is a service account tokens, no expiry handling. Both tools return their tab whole: the spreadsheet
is already summary-level, so there is nothing to filter client-side.

Structure - no class, no context manager, unlike the client side (mcp_client.py):
  - `server` (below) is one module-level MCPServer instance.
  - Each function decorated `@server.tool()` or `@server.prompt()` becomes
    visible to a connected client's list_tools()/list_prompts(); everything
    else (_init, _load_config, _read_tab, ...) is a private helper, invisible
    to the client.
  - `if __name__ == "__main__":` calls `_init()` once, then
    `asyncio.run(server.run_stdio_async())`, which blocks for the process's
    entire life: read a request from stdin, dispatch it to the matching
    decorated function, write the response to stdout, repeat. There's no
    __aenter__/__aexit__ pair here because the server doesn't initiate a
    connection - it just runs until stdin closes, which happens when the
    client's __aexit__ closes its end of the pipe.
"""
import asyncio

import contextlib
import io
import os
import shutil
import tempfile
from pathlib import Path

from main import run_accum
from models import resolve_config_path
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


def _init() -> None:
    """
    Load config and authenticate, populating the module-level Sheets API handle.
    Kept out of module scope - and called only from `__main__` - so importing
    this module (e.g. to test it, or for a clone with no config.toml yet) never
    touches the filesystem, credentials, or the network.
    """

@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
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
      verbose: bool = False,
  ) -> dict:
      """Does accumulation reach the income-formula bar by the plan date?
      Runs the S1 accumulation projection and returns the P50/tail portfolio
      at retirement, required stock CAGR, and the required-portfolio bar."""
      overrides = {k: v for k, v in locals().items()
                   if k in _ACCUM_WHITELIST and v is not None}

      prev_fire_config = os.environ.get("FIRE_CONFIG")
      tmp_dir_ctx = tempfile.TemporaryDirectory(prefix="fire_mcp_") if overrides else None
      try:
          if overrides:
              tmp_config = Path(tmp_dir_ctx.name) / "config.toml"
              shutil.copy(Path(resolve_config_path()), tmp_config)
              error = apply_cli_overrides(tmp_config, overrides)
              if error:
                  raise ValueError(error)
              os.environ["FIRE_CONFIG"] = str(tmp_config)

          buf = io.StringIO()
          with contextlib.redirect_stdout(buf):
              text, p50, tail, required_cagr, tail_pct, accum_required_it, _verdict_inputs = (
                  run_accum(verbose=verbose, planning_cagr=cagr)
              )
      finally:
          if overrides:
              if prev_fire_config is None:
                  os.environ.pop("FIRE_CONFIG", None)
              else:
                  os.environ["FIRE_CONFIG"] = prev_fire_config
              tmp_dir_ctx.cleanup()

      return {
          "p50_portfolio_eur": p50,
          "tail_portfolio_eur": tail,
          "tail_percentile": tail_pct,
          "required_stock_cagr": required_cagr,
          "required_portfolio_income_bar_eur": accum_required_it,
          "report": text,
      }

@server.prompt()
def projection() -> str:
    """Portfolio lifecycle projection only: run the portfolio lifecycle simulation and return a summary."""
    return (
        "Give me a portfolio-lifecycle-analysis review: current situation, plan and risks."
    )


if __name__ == "__main__":
    _init()
    asyncio.run(server.run_stdio_async())
