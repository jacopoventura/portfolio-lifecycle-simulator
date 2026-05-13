import os


def format_currency(value: float) -> str:
    return f"{value:,.2f}"


def format_currency_signed(value: float) -> str:
    return f"{value:+,.2f}"


def format_ratio_as_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_percent_value(value: float) -> str:
    return f"{value:.2f}%"


def format_ratio_delta_as_percentage_points(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _print_section(number: int, title: str) -> None:
    print("\n")
    print(f"  {number}. {title}")
    print()


def _print_subsection(title: str) -> None:
    print(f"\n\n  {'─' * 66}")
    print(f"  {title}")
    print(f"  {'─' * 66}\n")
