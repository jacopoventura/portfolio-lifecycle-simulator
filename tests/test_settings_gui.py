"""Tests for settings_gui's pure functions (no tkinter needed)."""

import tomllib
from pathlib import Path

import pytest

from settings_gui import _FIELDS, _validate, replace_toml_values

_SAMPLE = """# Scenario
target_monthly_retirement_income = 3000.0  # monthly income in retirement (today's money)
years = 16   # investment horizon in years
expected_inflation_percent = 2.3
bonds_percentage = 50.0

# Current Assets
current_stock_value = 50000
annual_net_saving = 24000.0
stock_monthly_investment_pct = 0.6
"""


class TestReplaceTomlValues:
    def test_replaces_value_and_keeps_inline_comment(self):
        out = replace_toml_values(_SAMPLE, {"years": 20})
        assert "years = 20   # investment horizon in years" in out

    def test_replaces_float(self):
        out = replace_toml_values(_SAMPLE, {"expected_inflation_percent": 2.5})
        assert "expected_inflation_percent = 2.5" in out

    def test_integral_float_written_without_decimal(self):
        out = replace_toml_values(_SAMPLE, {"target_monthly_retirement_income": 3500.0})
        assert "target_monthly_retirement_income = 3500  # monthly income" in out

    def test_untouched_lines_identical(self):
        out = replace_toml_values(_SAMPLE, {"years": 20})
        for line_in, line_out in zip(_SAMPLE.splitlines(), out.splitlines()):
            if not line_in.startswith("years"):
                assert line_in == line_out

    def test_result_parses_with_new_values(self):
        out = replace_toml_values(
            _SAMPLE, {"years": 18, "current_stock_value": 150000, "stock_monthly_investment_pct": 0.55}
        )
        parsed = tomllib.loads(out)
        assert parsed["years"] == 18
        assert parsed["current_stock_value"] == 150000
        assert parsed["stock_monthly_investment_pct"] == 0.55

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            replace_toml_values(_SAMPLE, {"nonexistent_key": 1.0})

    def test_all_gui_fields_exist_in_real_config(self):
        raw = (Path(__file__).parent.parent / "config.toml").read_text()
        parsed = tomllib.loads(raw)
        for key, _label, _unit in _FIELDS:
            assert key in parsed, f"GUI field {key} missing from config.toml"
            # Round-trip each field through the writer at its current value.
            out = replace_toml_values(raw, {key: float(parsed[key])})
            assert tomllib.loads(out)[key] == pytest.approx(float(parsed[key]))


class TestValidate:
    def _values(self, **overrides) -> dict:
        base = {
            "target_monthly_retirement_income": 3000.0,
            "years": 16.0,
            "expected_yearly_growth_stocks_percent": 6.0,
            "expected_inflation_percent": 2.3,
            "bonds_percentage": 50.0,
            "yield_bonds": 3.1,
            "withdrawal_percentage_stock": 2.8,
            "current_stock_value": 50000.0,
            "current_bonds_value": 10000.0,
            "current_cash": 30000.0,
            "annual_net_saving": 24000.0,
            "stock_monthly_investment_pct": 0.6,
            "bonds_monthly_investment_pct": 0.25,
        }
        base.update(overrides)
        return base

    def test_valid_config_passes(self):
        assert _validate(self._values()) is None

    def test_negative_years_rejected(self):
        assert _validate(self._values(years=0)) is not None

    def test_negative_asset_rejected(self):
        assert _validate(self._values(current_cash=-1)) is not None

    def test_bonds_percentage_bounds(self):
        assert _validate(self._values(bonds_percentage=101)) is not None

    def test_bond_yield_floor(self):
        assert _validate(self._values(yield_bonds=-100)) is not None

    def test_withdrawal_rate_must_be_positive(self):
        assert _validate(self._values(withdrawal_percentage_stock=0)) is not None

    def test_saving_shares_must_not_exceed_one(self):
        assert _validate(self._values(
            stock_monthly_investment_pct=0.7, bonds_monthly_investment_pct=0.4
        )) is not None

    def test_saving_shares_sum_to_one_ok(self):
        assert _validate(self._values(
            stock_monthly_investment_pct=0.7, bonds_monthly_investment_pct=0.3
        )) is None
