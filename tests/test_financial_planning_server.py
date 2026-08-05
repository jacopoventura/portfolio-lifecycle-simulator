"""Tests for the financial planning MCP server (financial_planning_server.py).

Runs against config.example.toml via FIRE_CONFIG so it never touches the real
config.toml. Tool functions are called directly, not through the stdio
transport - mirrors ../AIagent/tests/test_sheets_server.py's approach for its
sibling server.
"""
import asyncio
import os
from pathlib import Path

import pytest

import financial_planning_server as server_module
from financial_planning_server import accumulation_projection

_EXAMPLE_CONFIG = Path(__file__).resolve().parent.parent / "config.example.toml"


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch):
    """Point FIRE_CONFIG at config.example.toml for every test in this file."""
    monkeypatch.setenv("FIRE_CONFIG", str(_EXAMPLE_CONFIG))


class TestServerRegistersTools:
    def test_lists_accumulation_projection(self):
        tools = asyncio.run(server_module.server.list_tools())
        assert {t.name for t in tools} == {"accumulation_projection"}

    def test_lists_projection_prompt(self):
        prompts = asyncio.run(server_module.server.list_prompts())
        assert {p.name for p in prompts} == {"projection"}


class TestAccumulationProjection:
    def test_returns_structured_numbers_with_working_inputs(self):
        result = accumulation_projection(
            current_stock_value=100_000, current_bonds_value=10_000,
            current_cash=50_000, annual_net_saving=24_000,
        )
        assert result["p50_portfolio_eur"] > 0
        assert 0 < result["tail_portfolio_eur"] <= result["p50_portfolio_eur"]
        assert result["required_portfolio_income_bar_eur"] > 0

    def test_report_omitted_by_default(self):
        result = accumulation_projection(
            current_stock_value=100_000, current_bonds_value=10_000,
            current_cash=50_000, annual_net_saving=24_000,
        )
        assert "report" not in result

    def test_report_included_when_requested(self):
        result = accumulation_projection(
            current_stock_value=100_000, current_bonds_value=10_000,
            current_cash=50_000, annual_net_saving=24_000, include_report=True,
        )
        assert "TAKE AWAY" in result["report"]

    def test_zeroed_base_config_fails_cleanly(self):
        # config.example.toml ships with all "<-- EDIT ME" zeros - the
        # underlying pipeline should reject it, not raise or hang.
        result = accumulation_projection()
        assert result["p50_portfolio_eur"] == 0.0
        assert result["required_stock_cagr"] is None

    def test_invalid_override_raises(self):
        with pytest.raises(ValueError, match="exceed 1"):
            accumulation_projection(
                stock_monthly_investment_pct=0.9,
                bonds_monthly_investment_pct=0.5,
            )

    def test_does_not_leak_fire_config_env_var(self):
        prev = os.environ.get("FIRE_CONFIG")
        accumulation_projection(
            current_stock_value=100_000, current_bonds_value=10_000,
            current_cash=50_000, annual_net_saving=24_000,
        )
        assert os.environ.get("FIRE_CONFIG") == prev

    def test_does_not_write_real_config(self):
        real_config = Path(__file__).resolve().parent.parent / "config.toml"
        before = real_config.read_bytes() if real_config.exists() else None
        accumulation_projection(
            current_stock_value=100_000, current_bonds_value=10_000,
            current_cash=50_000, annual_net_saving=24_000,
        )
        after = real_config.read_bytes() if real_config.exists() else None
        assert before == after
