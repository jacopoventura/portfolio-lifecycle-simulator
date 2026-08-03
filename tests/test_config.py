"""Tests for Config.from_dict's fast_mc scaling (dev/iteration speed toggle)
and resolve_config_path's FIRE_CONFIG isolation (AI-agent / exploratory runs)."""

import os

from models import Config, resolve_config_path


class TestResolveConfigPath:
    def test_defaults_to_config_toml_when_unset(self, monkeypatch):
        monkeypatch.delenv("FIRE_CONFIG", raising=False)
        assert resolve_config_path() == "config.toml"

    def test_honors_custom_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("FIRE_CONFIG", raising=False)
        assert resolve_config_path("/some/other/config.toml") == "/some/other/config.toml"

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("FIRE_CONFIG", "/tmp/isolated/config.toml")
        assert resolve_config_path() == "/tmp/isolated/config.toml"
        assert resolve_config_path("config.toml") == "/tmp/isolated/config.toml"


def _raw(**overrides) -> dict:
    base = {
        "current_stock_value": 100_000.0,
        "target_monthly_retirement_income": 3_000.0,
        "years": 16.0,
        "expected_inflation_percent": 2.3,
        "expected_yearly_growth_stocks_percent": 6.0,
        "bonds_percentage": 50.0,
        "yield_bonds": 3.1,
        "tax_rate": 0.26,
        "withdrawal_percentage_stock": 2.8,
    }
    base.update(overrides)
    return base


class TestFastMcDefault:
    def test_defaults_to_false(self):
        config = Config.from_dict(_raw())
        assert config.fast_mc is False

    def test_n_simulations_unscaled_when_off(self):
        config = Config.from_dict(_raw(mc_n_simulations=50_000, mc_decumulation_n_simulations=10_000))
        assert config.mc_n_simulations == 50_000
        assert config.mc_decumulation_n_simulations == 10_000

    def test_grid_step_unscaled_when_off(self):
        config = Config.from_dict(_raw(mc_grid_cagr_step=0.01, optimize_grid_step=50_000.0))
        assert config.mc_grid_cagr_step == 0.01
        assert config.optimize_grid_step == 50_000.0


class TestFastMcOn:
    def test_flag_set(self):
        config = Config.from_dict(_raw(fast_mc=True))
        assert config.fast_mc is True

    def test_n_simulations_scaled_to_10_percent(self):
        config = Config.from_dict(_raw(
            fast_mc=True,
            mc_n_simulations=50_000,
            mc_decumulation_n_simulations=10_000,
            mc_grid_n_simulations=15_000,
        ))
        assert config.mc_n_simulations == 5_000
        assert config.mc_decumulation_n_simulations == 1_000
        assert config.mc_grid_n_simulations == 1_500

    def test_n_simulations_floored_at_100(self):
        # 10% of 500 would be 50 — too few sims to be meaningful, floor at 100.
        config = Config.from_dict(_raw(fast_mc=True, mc_n_simulations=500))
        assert config.mc_n_simulations == 100

    def test_grid_steps_scaled_10x_coarser(self):
        config = Config.from_dict(_raw(
            fast_mc=True, mc_grid_cagr_step=0.01, optimize_grid_step=50_000.0,
        ))
        assert config.mc_grid_cagr_step == 0.1
        assert config.optimize_grid_step == 500_000.0

    def test_non_mc_fields_untouched(self):
        # fast_mc must not leak into unrelated fields.
        config = Config.from_dict(_raw(fast_mc=True, years=16.0, bonds_percentage=50.0))
        assert config.years == 16.0
        assert config.bonds_percentage == 50.0
