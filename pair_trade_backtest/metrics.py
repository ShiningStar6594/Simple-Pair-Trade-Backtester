from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from pair_trade_backtest import BacktestResult, Trade

_SQRT_252 = sqrt(252.0)


@dataclass(frozen=True)
class WindowStats:
    n_admitted_pairs: int
    n_trades: int
    total_pnl: float
    hit_rate: float
    mean_trade_pnl: float
    median_trade_pnl: float
    share_zscore_target: float
    share_statistical_exit: float
    share_period_end: float
    sharpe: float
    max_drawdown: float


@dataclass(frozen=True)
class PairRow:
    ticker_a: str
    ticker_b: str
    n_trades: int
    total_pnl: float
    hit_rate: float


def daily_realized_pnl(trades: list[Trade]) -> pd.Series:
    buckets: dict[pd.Timestamp, float] = {}
    for trade in trades:
        if trade.exit_date is None:
            continue
        day = pd.Timestamp(trade.exit_date)
        buckets[day] = buckets.get(day, 0.0) + trade.pnl
    if not buckets:
        return pd.Series(dtype=float)
    return pd.Series(buckets, dtype=float).sort_index()


def equity_curve(trades: list[Trade]) -> pd.Series:
    daily = daily_realized_pnl(trades)
    if daily.empty:
        return daily
    return daily.cumsum()


def summarize_window(
    trades: list[Trade],
    *,
    n_admitted_pairs: int = 0,
) -> WindowStats:
    n = len(trades)
    if n == 0:
        return WindowStats(
            n_admitted_pairs=n_admitted_pairs,
            n_trades=0,
            total_pnl=0.0,
            hit_rate=0.0,
            mean_trade_pnl=0.0,
            median_trade_pnl=0.0,
            share_zscore_target=0.0,
            share_statistical_exit=0.0,
            share_period_end=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
        )
    pnls = np.array([t.pnl for t in trades], dtype=float)
    reasons = [t.exit_reason for t in trades]
    daily = daily_realized_pnl(trades)
    return WindowStats(
        n_admitted_pairs=n_admitted_pairs,
        n_trades=n,
        total_pnl=float(pnls.sum()),
        hit_rate=float(np.mean(pnls > 0)),
        mean_trade_pnl=float(np.mean(pnls)),
        median_trade_pnl=float(np.median(pnls)),
        share_zscore_target=reasons.count("zscore_target") / n,
        share_statistical_exit=reasons.count("statistical_exit") / n,
        share_period_end=reasons.count("period_end") / n,
        sharpe=_sharpe(daily),
        max_drawdown=_max_drawdown(daily.cumsum()),
    )


def summarize_result(result: BacktestResult) -> WindowStats:
    return summarize_window(result.trades, n_admitted_pairs=len(result.admitted_pairs))


def pick_review_pair(trades: list[Trade]) -> tuple[str, str] | None:
    rows = pair_rows(trades, cap=10**9)
    if not rows:
        return None
    worst = rows[-1]
    return worst.ticker_a, worst.ticker_b


def format_window_block(title: str, stats: WindowStats) -> str:
    return "\n".join(
        [
            title,
            (
                f"admitted {stats.n_admitted_pairs}  trades {stats.n_trades}  "
                f"total_pnl {stats.total_pnl:.4f}  hit_rate {stats.hit_rate:.3f}"
            ),
            (
                f"mean_trade {stats.mean_trade_pnl:.4f}  "
                f"median_trade {stats.median_trade_pnl:.4f}"
            ),
            (
                f"exits zscore_target {stats.share_zscore_target:.3f}  "
                f"statistical_exit {stats.share_statistical_exit:.3f}  "
                f"period_end {stats.share_period_end:.3f}"
            ),
            f"sharpe {stats.sharpe:.3f}  max_drawdown {stats.max_drawdown:.4f}",
        ]
    )


def format_pair_table(rows: list[PairRow]) -> str:
    lines = ["pair              n_trades  total_pnl  hit_rate"]
    for row in rows:
        name = f"{row.ticker_a}/{row.ticker_b}"
        lines.append(
            f"{name:16s}  {row.n_trades:8d}  {row.total_pnl:9.4f}  {row.hit_rate:8.3f}"
        )
    return "\n".join(lines)


def pair_rows(trades: list[Trade], cap: int = 15) -> list[PairRow]:
    grouped: dict[tuple[str, str], list[Trade]] = defaultdict(list)
    for trade in trades:
        grouped[(trade.ticker_a, trade.ticker_b)].append(trade)
    rows = [
        PairRow(
            ticker_a=a,
            ticker_b=b,
            n_trades=len(items),
            total_pnl=float(sum(t.pnl for t in items)),
            hit_rate=(
                sum(1 for t in items if t.pnl > 0) / len(items) if items else 0.0
            ),
        )
        for (a, b), items in grouped.items()
    ]
    rows.sort(key=lambda row: row.total_pnl, reverse=True)
    if len(rows) <= cap:
        return rows
    n_top = (cap + 1) // 2
    n_bottom = cap - n_top
    top = rows[:n_top]
    bottom = rows[-n_bottom:]
    seen = {(r.ticker_a, r.ticker_b) for r in top}
    extra = [r for r in bottom if (r.ticker_a, r.ticker_b) not in seen]
    return top + extra


def _sharpe(daily: pd.Series) -> float:
    if len(daily) < 2:
        return 0.0
    std = float(daily.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(daily.mean() / std * _SQRT_252)


def _max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return 0.0
    peak = curve.cummax()
    return float((peak - curve).max())
