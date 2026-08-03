"""Plan-lever settings — either an interactive dialog shown by main.py before
the pipeline runs, or non-interactive overrides for scripted / AI-agent use
(main.py --<param> flags, generated from GUI_FIELDS below skip the dialog).

Exposes only the plan levers (portfolio, saving, horizon, target income,
CAGR/inflation, bonds split) — config.toml stays the source of truth for
everything else. Saving preserves the TOML file byte-for-byte except for the
edited values (comments and formatting untouched).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


# (toml key, label, unit hint) — also drives main.py's --<param> CLI flags.
GUI_FIELDS: list[tuple[str, str, str]] = [
    ("target_monthly_retirement_income", "Target retirement income", "€/mo today's €"),
    ("years", "Years to retirement", "y"),
    ("expected_yearly_growth_stocks_percent", "Planning stock CAGR", "%/yr net of TER"),
    ("expected_inflation_percent", "Expected inflation", "%/yr"),
    ("bonds_percentage", "Bonds at retirement", "% of portfolio"),
    ("yield_bonds", "Bond yield", "%/yr"),
    ("withdrawal_percentage_stock", "Stock withdrawal rate", "%/yr"),
    ("current_stock_value", "Stocks today", "€"),
    ("current_bonds_value", "Bonds today", "€"),
    ("current_cash", "Cash today", "€"),
    ("annual_net_saving", "Annual net saving", "€/yr"),
    ("stock_monthly_investment_pct", "Saving share → stocks", "0–1"),
    ("bonds_monthly_investment_pct", "Saving share → bonds", "0–1"),
    ("mc_decumulation_years", "Decumulation horizon", "y"),
    ("fi_preservation_ratio", "Preservation ratio", "0–1 (fraction of capital)"),
    ("fi_preservation_confidence_baseline", "Preservation confidence (baseline)", "0–1"),
    ("fi_preservation_confidence_stress", "Preservation confidence (worst-SORR)", "0–1"),
]

_SECTIONS: list[tuple[str, list[str]]] = [
    ("Scenario", ["target_monthly_retirement_income", "years",
                  "expected_yearly_growth_stocks_percent",
                  "expected_inflation_percent", "bonds_percentage",
                  "yield_bonds", "withdrawal_percentage_stock"]),
    ("Current assets", ["current_stock_value", "current_bonds_value", "current_cash"]),
    ("Saving", ["annual_net_saving", "stock_monthly_investment_pct",
                "bonds_monthly_investment_pct"]),
    ("Preservation", ["mc_decumulation_years", "fi_preservation_ratio",
                       "fi_preservation_confidence_baseline",
                       "fi_preservation_confidence_stress"]),
]


def _format_number(v: float) -> str:
    """Render integral floats without a trailing .0 so the TOML stays tidy."""
    return str(int(v)) if float(v) == int(v) else repr(float(v))


def replace_toml_values(text: str, updates: dict[str, float]) -> str:
    """Replace top-level `key = value` lines, preserving inline comments and
    everything else byte-for-byte. Raises KeyError if a key is missing."""
    for key, value in updates.items():
        pattern = re.compile(
            rf"(?m)^({re.escape(key)}\s*=\s*)([^#\n]*?)(\s*#.*)?$"
        )
        new_text, n = pattern.subn(
            lambda m: f"{m.group(1)}{_format_number(value)}{m.group(3) or ''}",
            text,
            count=1,
        )
        if n == 0:
            raise KeyError(f"config key not found: {key}")
        text = new_text
    return text


def _validate(values: dict[str, float]) -> str | None:
    """Return an error message, or None if the plan levers are consistent."""
    if values["years"] <= 0:
        return "Years to retirement must be positive."
    for key in ("current_stock_value", "current_bonds_value", "current_cash",
                "annual_net_saving", "target_monthly_retirement_income"):
        if values[key] < 0:
            return f"{key} cannot be negative."
    if not 0 <= values["bonds_percentage"] <= 100:
        return "Bonds at retirement must be between 0 and 100."
    if values["yield_bonds"] <= -100:
        return "Bond yield must be greater than -100."
    if values["withdrawal_percentage_stock"] <= 0:
        return "Stock withdrawal rate must be greater than 0."
    s = values["stock_monthly_investment_pct"]
    b = values["bonds_monthly_investment_pct"]
    if not (0 <= s <= 1 and 0 <= b <= 1):
        return "Saving shares must be between 0 and 1."
    if s + b > 1.0 + 1e-9:
        return "Stock + bond saving shares exceed 1 (the rest goes to cash)."
    if values["mc_decumulation_years"] <= 0:
        return "Decumulation horizon must be positive."
    if not 0 <= values["fi_preservation_ratio"] <= 1:
        return "Preservation ratio must be between 0 and 1."
    if not 0 <= values["fi_preservation_confidence_baseline"] <= 1:
        return "Preservation confidence (baseline) must be between 0 and 1."
    if not 0 <= values["fi_preservation_confidence_stress"] <= 1:
        return "Preservation confidence (worst-SORR) must be between 0 and 1."
    return None


def apply_cli_overrides(config_path: Path, overrides: dict[str, float]) -> str | None:
    """Non-interactive equivalent of the GUI dialog: write `overrides` (a
    subset of GUI_FIELDS keys) into config.toml, validated the same way.

    Returns an error message if the resulting plan levers are inconsistent
    (nothing is written in that case), or None on success. No-op if
    `overrides` is empty. Used by main.py when called with plan-parameter
    flags (e.g. from an AI agent) instead of the interactive dialog.
    """
    if not overrides:
        return None
    raw = config_path.read_text()
    parsed = tomllib.loads(raw)
    values = {key: float(parsed[key]) for key, _label, _unit in GUI_FIELDS}
    values.update(overrides)
    error = _validate(values)
    if error:
        return error
    config_path.write_text(replace_toml_values(raw, overrides))
    return None


def edit_settings(config_path: Path) -> bool:
    """Show the plan-lever dialog. Returns True to proceed with the run
    (saving config.toml if values changed), False if the user cancelled."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    raw = config_path.read_text()
    config = tomllib.loads(raw)

    root = tk.Tk()
    root.title("Retirement plan — parameters")
    root.resizable(False, False)

    result = {"run": False}
    entries: dict[str, tk.StringVar] = {}
    labels = {key: (label, unit) for key, label, unit in GUI_FIELDS}

    outer = ttk.Frame(root, padding=12)
    outer.grid(sticky="nsew")

    for section, keys in _SECTIONS:
        frame = ttk.LabelFrame(outer, text=section, padding=8)
        frame.pack(fill="x", pady=(0, 8))
        for row, key in enumerate(keys):
            label, unit = labels[key]
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8))
            var = tk.StringVar(value=_format_number(float(config[key])))
            entries[key] = var
            ttk.Entry(frame, textvariable=var, width=12, justify="right").grid(
                row=row, column=1, sticky="e"
            )
            ttk.Label(frame, text=unit, foreground="gray").grid(
                row=row, column=2, sticky="w", padx=(6, 0)
            )
        frame.columnconfigure(0, weight=1)

    # Live preview of the monthly saving split.
    split_label = ttk.Label(outer, foreground="gray")
    split_label.pack(anchor="w", pady=(0, 8))

    def _refresh_split(*_args: object) -> None:
        try:
            saving = float(entries["annual_net_saving"].get()) / 12.0
            s = saving * float(entries["stock_monthly_investment_pct"].get())
            b = saving * float(entries["bonds_monthly_investment_pct"].get())
            c = max(0.0, saving - s - b)
            split_label.config(
                text=f"Monthly saving split — stocks €{s:,.0f} · bonds €{b:,.0f} · cash €{c:,.0f}"
            )
        except ValueError:
            split_label.config(text="Monthly saving split — (enter valid numbers)")

    for key in ("annual_net_saving", "stock_monthly_investment_pct",
                "bonds_monthly_investment_pct"):
        entries[key].trace_add("write", _refresh_split)
    _refresh_split()

    def _on_run() -> None:
        try:
            values = {key: float(var.get()) for key, var in entries.items()}
        except ValueError:
            messagebox.showerror("Invalid input", "All fields must be numbers.", parent=root)
            return
        error = _validate(values)
        if error:
            messagebox.showerror("Invalid input", error, parent=root)
            return
        changed = {k: v for k, v in values.items() if float(config[k]) != v}
        if changed:
            config_path.write_text(replace_toml_values(raw, changed))
        result["run"] = True
        root.destroy()

    def _on_cancel() -> None:
        root.destroy()

    buttons = ttk.Frame(outer)
    buttons.pack(fill="x")
    ttk.Button(buttons, text="Run simulation", command=_on_run).pack(side="right")
    ttk.Button(buttons, text="Cancel", command=_on_cancel).pack(side="right", padx=(0, 8))

    root.protocol("WM_DELETE_WINDOW", _on_cancel)
    root.bind("<Return>", lambda _e: _on_run())
    root.bind("<Escape>", lambda _e: _on_cancel())
    root.eval("tk::PlaceWindow . center")
    root.mainloop()
    return result["run"]
