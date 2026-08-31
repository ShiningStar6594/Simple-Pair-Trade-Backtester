from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pair_trade_backtest import Trade
from pair_trade_backtest.metrics import equity_curve


def spread_and_zscore(
    close_a: pd.Series,
    close_b: pd.Series,
    intercept: float,
    hedge: float,
    zscore_window: int,
) -> tuple[pd.Series, pd.Series]:
    spread = np.log(close_a) - intercept - hedge * np.log(close_b)
    spread = pd.Series(spread, index=close_a.index)
    rolling = spread.rolling(zscore_window)
    roll_std = rolling.std(ddof=0)
    zscore = (spread - rolling.mean()) / roll_std
    zscore = zscore.where(roll_std > 1e-8)
    return spread, zscore


def save_equity_curve(
    trades: list[Trade],
    path: Path,
    title: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    curve = equity_curve(trades)
    fig, ax = plt.subplots(figsize=(10, 4))
    if curve.empty:
        ax.set_title(title + " (no trades)")
    else:
        ax.plot(curve.index, curve.to_numpy(), color="black", linewidth=1.2)
        ax.set_title(title)
    ax.set_ylabel("Cumulative PnL after costs ($1 of A book)")
    ax.set_xlabel("Exit date")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_pair_path(
    spread: pd.Series,
    zscore: pd.Series,
    trades: list[Trade],
    path: Path,
    title: str,
    zscore_entry: float = 2.0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_s, ax_z) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_s.plot(spread.index, spread.to_numpy(), color="black", linewidth=1.0)
    ax_s.set_ylabel("Spread")
    ax_s.set_title(title)
    ax_z.plot(zscore.index, zscore.to_numpy(), color="black", linewidth=1.0)
    ax_z.axhline(zscore_entry, color="gray", linestyle="--", linewidth=0.8)
    ax_z.axhline(-zscore_entry, color="gray", linestyle="--", linewidth=0.8)
    ax_z.axhline(0.0, color="gray", linewidth=0.8)
    ax_z.set_ylabel("Z-Score")
    ax_z.set_xlabel("Date")
    for trade in trades:
        ax_s.axvline(trade.fill_date, color="tab:blue", alpha=0.5, linewidth=0.8)
        ax_z.axvline(trade.fill_date, color="tab:blue", alpha=0.5, linewidth=0.8)
        if trade.exit_date is not None:
            ax_s.axvline(trade.exit_date, color="tab:orange", alpha=0.5, linewidth=0.8)
            ax_z.axvline(trade.exit_date, color="tab:orange", alpha=0.5, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
