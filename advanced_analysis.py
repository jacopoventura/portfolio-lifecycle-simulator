from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from random import Random

import tomllib

from main import Config, validate_config_inputs, validate_projection_inputs


@dataclass
class MonteCarloSettings:
    simulations: int
    random_seed: int | None
    stock_volatility: float
    inflation_volatility: float
    bond_yield_volatility: float

    @classmethod
    def from_dict(cls, config: dict[str, object]) -> "MonteCarloSettings":
        seed_raw = config.get("monte_carlo_random_seed")
        random_seed = int(seed_raw) if seed_raw is not None else None
        return cls(
            simulations=int(config.get("monte_carlo_simulations", 10_000)),
            random_seed=random_seed,
            stock_volatility=float(config.get("stock_volatility_percent", 18.0)) / 100,
            inflation_volatility=float(config.get("inflation_volatility_percent", 1.0)) / 100,
            bond_yield_volatility=float(config.get("bond_yield_volatility_percent", 0.7)) / 100,
        )

    def with_overrides(self, args: argparse.Namespace) -> "MonteCarloSettings":
        return MonteCarloSettings(
            simulations=args.simulations if args.simulations is not None else self.simulations,
            random_seed=args.seed if args.seed is not None else self.random_seed,
            stock_volatility=(
                args.stock_volatility_percent / 100
                if args.stock_volatility_percent is not None
                else self.stock_volatility
            ),
            inflation_volatility=(
                args.inflation_volatility_percent / 100
                if args.inflation_volatility_percent is not None
                else self.inflation_volatility
            ),
            bond_yield_volatility=(
                args.bond_yield_volatility_percent / 100
                if args.bond_yield_volatility_percent is not None
                else self.bond_yield_volatility
            ),
        )


@dataclass
class PathOutcome:
    final_portfolio_value: float
    target_annual_net_income: float
    annual_net_income_scenario_1: float
    annual_net_income_scenario_2: float


@dataclass
class DistributionStats:
    mean: float
    p05: float
    p50: float
    p95: float


@dataclass
class ScenarioMonteCarloSummary:
    name: str
    success_probability: float
    annual_net_income: DistributionStats
    annual_shortfall: DistributionStats


@dataclass
class MonteCarloResult:
    simulations: int
    years: float
    random_seed: int | None
    portfolio_stats: DistributionStats
    target_income_stats: DistributionStats
    scenario_1: ScenarioMonteCarloSummary
    scenario_2: ScenarioMonteCarloSummary
    net_income_delta_s2_s1: DistributionStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advanced Monte Carlo analysis for retirement income outcomes.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.toml"),
        help="Path to TOML configuration file.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="Override number of Monte Carlo simulations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for reproducibility.",
    )
    parser.add_argument(
        "--stock-volatility-percent",
        type=float,
        default=None,
        help="Override annual stock return volatility in percent (for example 18).",
    )
    parser.add_argument(
        "--inflation-volatility-percent",
        type=float,
        default=None,
        help="Override annual inflation volatility in percent (for example 1).",
    )
    parser.add_argument(
        "--bond-yield-volatility-percent",
        type=float,
        default=None,
        help="Override annual bond-yield volatility in percent at horizon (for example 0.7).",
    )
    return parser.parse_args()


def validate_settings(settings: MonteCarloSettings) -> None:
    if settings.simulations <= 0:
        raise ValueError("monte_carlo_simulations must be greater than 0.")
    if settings.stock_volatility < 0:
        raise ValueError("stock_volatility_percent must be greater than or equal to 0.")
    if settings.inflation_volatility < 0:
        raise ValueError("inflation_volatility_percent must be greater than or equal to 0.")
    if settings.bond_yield_volatility < 0:
        raise ValueError("bond_yield_volatility_percent must be greater than or equal to 0.")


def clamp_rate(annual_rate: float) -> float:
    return max(annual_rate, -0.999_999)


def monthly_rate_from_annual(annual_rate: float) -> float:
    return (1 + annual_rate) ** (1 / 12) - 1


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty data.")
    if not 0 <= q <= 1:
        raise ValueError("Percentile q must be between 0 and 1.")

    pos = (len(sorted_values) - 1) * q
    low_index = math.floor(pos)
    high_index = math.ceil(pos)
    if low_index == high_index:
        return sorted_values[low_index]
    low_value = sorted_values[low_index]
    high_value = sorted_values[high_index]
    return low_value + (high_value - low_value) * (pos - low_index)


def summarize_distribution(values: list[float]) -> DistributionStats:
    ordered = sorted(values)
    return DistributionStats(
        mean=statistics.fmean(ordered),
        p05=percentile(ordered, 0.05),
        p50=percentile(ordered, 0.50),
        p95=percentile(ordered, 0.95),
    )


def simulate_path(
    config: Config,
    settings: MonteCarloSettings,
    rng: Random,
    stock_expected_yield: float,
    inflation_mean: float,
    bonds_yield_mean: float,
    withdrawal_rate: float,
) -> PathOutcome:
    months = int(round(config.years * 12))
    yearly_steps = math.ceil(months / 12)

    cash_interest_rate = config.interest_on_cash_pa / 100
    cash_monthly_rate = monthly_rate_from_annual(cash_interest_rate)

    initial_cash_allocated_to_stock = config.current_cash * config.pct_cash_to_be_allocated_in_stock_now
    stock_value = config.current_stock_value + initial_cash_allocated_to_stock
    cash_value = config.current_cash - initial_cash_allocated_to_stock

    monthly_net_saving = config.annual_net_saving / 12
    monthly_stock_contribution = monthly_net_saving * config.maximum_stock_monthly_investment_pct
    monthly_cash_contribution = monthly_net_saving - monthly_stock_contribution

    inflation_factor = 1.0
    months_done = 0

    for _ in range(yearly_steps):
        annual_stock_return = clamp_rate(
            rng.gauss(stock_expected_yield, settings.stock_volatility)
        )
        annual_inflation = clamp_rate(
            rng.gauss(inflation_mean, settings.inflation_volatility)
        )
        stock_monthly_rate = monthly_rate_from_annual(annual_stock_return)
        inflation_monthly_rate = monthly_rate_from_annual(annual_inflation)

        months_this_step = min(12, months - months_done)
        for _ in range(months_this_step):
            stock_value = stock_value * (1 + stock_monthly_rate) + monthly_stock_contribution
            cash_value = cash_value * (1 + cash_monthly_rate) + monthly_cash_contribution
            inflation_factor *= 1 + inflation_monthly_rate
        months_done += months_this_step

    final_cash_converted_to_bonds = cash_value * config.pct_final_cash_to_convert_to_bonds
    final_remaining_cash = cash_value - final_cash_converted_to_bonds
    final_bonds_value = final_cash_converted_to_bonds
    final_portfolio_value = stock_value + final_bonds_value + final_remaining_cash

    realized_bonds_yield = clamp_rate(
        rng.gauss(bonds_yield_mean, settings.bond_yield_volatility)
    )

    annual_net_income_stock = stock_value * withdrawal_rate * (1 - config.tax_rate)
    annual_net_income_bond_scenario_1 = (
        final_bonds_value * realized_bonds_yield * (1 - config.tax_rate)
    )
    annual_net_income_bond_scenario_2 = (
        final_bonds_value * realized_bonds_yield * (1 - config.tax_rate_bond_italy)
    )

    annual_net_income_scenario_1 = annual_net_income_stock + annual_net_income_bond_scenario_1
    annual_net_income_scenario_2 = annual_net_income_stock + annual_net_income_bond_scenario_2
    target_annual_net_income = config.income_value * inflation_factor * 12

    return PathOutcome(
        final_portfolio_value=final_portfolio_value,
        target_annual_net_income=target_annual_net_income,
        annual_net_income_scenario_1=annual_net_income_scenario_1,
        annual_net_income_scenario_2=annual_net_income_scenario_2,
    )


def run_monte_carlo(config: Config, settings: MonteCarloSettings) -> MonteCarloResult:
    stock_expected_yield = config.expected_yearly_growth_stocks_percent / 100
    inflation_mean = config.expected_inflation_percent / 100
    bonds_yield_mean = config.yield_bonds / 100
    withdrawal_rate = config.withdrawal_percentage_stock / 100

    validate_config_inputs(
        config=config,
        stock_expected_yield=stock_expected_yield,
        bonds_yield=bonds_yield_mean,
        withdrawal_rate=withdrawal_rate,
    )
    validate_projection_inputs(config, config.interest_on_cash_pa / 100)
    validate_settings(settings)

    rng = Random(settings.random_seed)
    final_portfolios: list[float] = []
    target_incomes: list[float] = []
    annual_incomes_s1: list[float] = []
    annual_incomes_s2: list[float] = []
    annual_shortfalls_s1: list[float] = []
    annual_shortfalls_s2: list[float] = []
    annual_deltas_s2_s1: list[float] = []
    success_count_s1 = 0
    success_count_s2 = 0

    for _ in range(settings.simulations):
        outcome = simulate_path(
            config=config,
            settings=settings,
            rng=rng,
            stock_expected_yield=stock_expected_yield,
            inflation_mean=inflation_mean,
            bonds_yield_mean=bonds_yield_mean,
            withdrawal_rate=withdrawal_rate,
        )
        final_portfolios.append(outcome.final_portfolio_value)
        target_incomes.append(outcome.target_annual_net_income)
        annual_incomes_s1.append(outcome.annual_net_income_scenario_1)
        annual_incomes_s2.append(outcome.annual_net_income_scenario_2)
        annual_shortfalls_s1.append(
            max(0.0, outcome.target_annual_net_income - outcome.annual_net_income_scenario_1)
        )
        annual_shortfalls_s2.append(
            max(0.0, outcome.target_annual_net_income - outcome.annual_net_income_scenario_2)
        )
        annual_deltas_s2_s1.append(
            outcome.annual_net_income_scenario_2 - outcome.annual_net_income_scenario_1
        )

        if outcome.annual_net_income_scenario_1 >= outcome.target_annual_net_income:
            success_count_s1 += 1
        if outcome.annual_net_income_scenario_2 >= outcome.target_annual_net_income:
            success_count_s2 += 1

    return MonteCarloResult(
        simulations=settings.simulations,
        years=config.years,
        random_seed=settings.random_seed,
        portfolio_stats=summarize_distribution(final_portfolios),
        target_income_stats=summarize_distribution(target_incomes),
        scenario_1=ScenarioMonteCarloSummary(
            name="SCENARIO 1",
            success_probability=success_count_s1 / settings.simulations,
            annual_net_income=summarize_distribution(annual_incomes_s1),
            annual_shortfall=summarize_distribution(annual_shortfalls_s1),
        ),
        scenario_2=ScenarioMonteCarloSummary(
            name="SCENARIO 2",
            success_probability=success_count_s2 / settings.simulations,
            annual_net_income=summarize_distribution(annual_incomes_s2),
            annual_shortfall=summarize_distribution(annual_shortfalls_s2),
        ),
        net_income_delta_s2_s1=summarize_distribution(annual_deltas_s2_s1),
    )


def format_currency(value: float) -> str:
    return f"{value:,.2f}"


def format_ratio_as_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_distribution_line(title: str, stats: DistributionStats) -> None:
    print(
        f"  {title:<38}"
        f"mean {format_currency(stats.mean):>14} | "
        f"p05 {format_currency(stats.p05):>14} | "
        f"p50 {format_currency(stats.p50):>14} | "
        f"p95 {format_currency(stats.p95):>14}"
    )


def print_report(config: Config, settings: MonteCarloSettings, result: MonteCarloResult) -> None:
    print("=" * 92)
    print("ADVANCED ANALYSIS: MONTE CARLO")
    print("=" * 92)
    print("Simulation assumptions")
    print(f"  Horizon: {result.years:.1f} years")
    print(f"  Simulations: {result.simulations:,}")
    print(f"  Seed: {result.random_seed if result.random_seed is not None else 'random'}")
    print(
        "  Stock return model (annual): "
        f"mean {config.expected_yearly_growth_stocks_percent:.2f}% "
        f"and vol {settings.stock_volatility * 100:.2f}%"
    )
    print(
        "  Inflation model (annual): "
        f"mean {config.expected_inflation_percent:.2f}% "
        f"and vol {settings.inflation_volatility * 100:.2f}%"
    )
    print(
        "  Bond yield model at horizon (annual): "
        f"mean {config.yield_bonds:.2f}% "
        f"and vol {settings.bond_yield_volatility * 100:.2f}%"
    )
    print()
    print("Path accumulation setup")
    print(f"  Initial stocks: {format_currency(config.current_stock_value)}")
    print(f"  Initial cash: {format_currency(config.current_cash)}")
    print(
        "  Cash allocated to stocks now: "
        f"{format_ratio_as_percent(config.pct_cash_to_be_allocated_in_stock_now)}"
    )
    print(f"  Annual net saving: {format_currency(config.annual_net_saving)}")
    print(
        "  Max monthly saving to stocks: "
        f"{format_ratio_as_percent(config.maximum_stock_monthly_investment_pct)}"
    )
    print(
        "  Final cash converted to bonds: "
        f"{format_ratio_as_percent(config.pct_final_cash_to_convert_to_bonds)}"
    )
    print()
    print("Distribution summary")
    print_distribution_line("Final portfolio value", result.portfolio_stats)
    print_distribution_line("Target annual net income", result.target_income_stats)
    print_distribution_line("Net income delta (S2-S1)", result.net_income_delta_s2_s1)
    print()
    print(f"{result.scenario_1.name}")
    print(
        "  Success probability (income >= target): "
        f"{format_ratio_as_percent(result.scenario_1.success_probability)}"
    )
    print_distribution_line("Annual net income", result.scenario_1.annual_net_income)
    print_distribution_line("Annual shortfall vs target", result.scenario_1.annual_shortfall)
    print()
    print(f"{result.scenario_2.name}")
    print(
        "  Success probability (income >= target): "
        f"{format_ratio_as_percent(result.scenario_2.success_probability)}"
    )
    print_distribution_line("Annual net income", result.scenario_2.annual_net_income)
    print_distribution_line("Annual shortfall vs target", result.scenario_2.annual_shortfall)


def load_inputs(config_path: Path, args: argparse.Namespace) -> tuple[Config, MonteCarloSettings]:
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    config = Config.from_dict(raw_config)
    settings = MonteCarloSettings.from_dict(raw_config).with_overrides(args)
    return config, settings


def main() -> None:
    args = parse_args()
    try:
        config, settings = load_inputs(args.config, args)
        result = run_monte_carlo(config, settings)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        return
    except KeyError as error:
        print(f"Missing required config key: {error}")
        return
    except (ValueError, tomllib.TOMLDecodeError) as error:
        print(error)
        return

    print_report(config, settings, result)


if __name__ == "__main__":
    main()
