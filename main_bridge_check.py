"""
Bridge check — how does the canonical FI required wealth move as you stack
individual config changes, one at a time?

Edit STEPS below to define which overrides to stack and in what order. Each
step's overrides are applied cumulatively on top of every previous step's, so
STEPS should read like a changelog: start from a known baseline, then layer on
one change per row.

Usage:
    python main_bridge_check.py
    python main_bridge_check.py --config config.toml --n-simulations 4000
"""
import argparse
import dataclasses
import tomllib

from models import Config, resolve_config_path
from monte_carlo import compute_canonical_fi_required_scan

# Example: the 2026-08-03 session bridge (old preservation params -> current).
# Replace with whatever comparison you want to run next.
STEPS: list[tuple[str, dict]] = [
    ("Original (before 2026-08-03 session)", dict(
        fi_preservation_ratio=0.66,
        fi_preservation_confidence_stress=0.50,
        target_monthly_retirement_income=3000.0,
    )),
    ("+ ratio 0.66 -> 0.75", dict(fi_preservation_ratio=0.75)),
    ("+ stress confidence 0.50 -> 0.80", dict(fi_preservation_confidence_stress=0.80)),
    ("+ spend correction 3000 -> 3100/mo", dict(target_monthly_retirement_income=3100.0)),
]


def _load_config(path: str | None = None) -> Config:
    with open(path or resolve_config_path(), "rb") as f:
        raw = tomllib.load(f)
    return Config.from_dict(raw)


def _print_table(rows: list[tuple[str, float, float, float, float, float]]) -> None:
    print()
    print(f"  {'Step':<40}  {'ratio':>6}  {'stress conf':>11}  {'spend/mo':>9}  "
          f"{'Required wealth':>16}  {'Allocation':>11}")
    print("  " + "-" * 100)
    for label, ratio, stress_conf, spend, required, bonds_pct in rows:
        print(f"  {label:<40}  {ratio:>5.2f}  {stress_conf * 100:>10.0f}%  "
              f"€{spend:>7,.0f}  €{required:>14,.0f}  {bonds_pct * 100:>3.0f}% bonds")
    print()


def main(
    *,
    config_path: str | None = None,
    n_simulations: int = 4000,
    hi: float = 15_000_000.0,
    seed: int = 42,
) -> None:
    base = _load_config(config_path)
    overrides: dict = {}
    rows = []

    for label, step_overrides in STEPS:
        overrides.update(step_overrides)
        cfg = dataclasses.replace(base, **overrides)
        infl_factor = (1.0 + cfg.expected_inflation_percent / 100.0) ** cfg.years
        target_monthly_nom = cfg.target_monthly_retirement_income * infl_factor
        scan = compute_canonical_fi_required_scan(
            cfg, target_monthly_nom,
            preserve_ratios=[cfg.fi_preservation_ratio],
            n_simulations=n_simulations, hi=hi, seed=seed,
        )
        r = scan.rows[0]
        rows.append((
            label, cfg.fi_preservation_ratio, cfg.fi_preservation_confidence_stress,
            cfg.target_monthly_retirement_income, r.required_wealth_eur, r.optimal_bonds_pct,
        ))

    _print_table(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Bridge check: stack config overrides and show the required-wealth impact of each"
    )
    p.add_argument("--config", default=None, help="Path to config file (default: FIRE_CONFIG env var, else config.toml)")
    p.add_argument("--n-simulations", type=int, default=4000)
    p.add_argument("--hi", type=float, default=15_000_000.0, help="Upper search bound in EUR")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(config_path=args.config, n_simulations=args.n_simulations, hi=args.hi, seed=args.seed)
