"""Walk-forward Z split. No live yfinance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pair_trade_backtest import BacktestConfig, PricePanel
from pair_trade_backtest.zscore_sweep import choose_zscore, split_walk_forward


def _panel(n: int) -> PricePanel:
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame({"AAA": np.linspace(10, 20, n), "BBB": np.linspace(20, 40, n)}, index=idx)
    return PricePanel(open=close.copy(), close=close)


def test_split_walk_forward_two_cycles_then_holdout():
    formation_days, trading_days = 252, 126
    cycle = formation_days + trading_days
    prices = _panel(3 * cycle)
    select, holdout = split_walk_forward(prices, formation_days, trading_days)
    assert len(select.close) == 2 * cycle
    assert len(holdout.close) >= cycle
    assert select.close.index[-1] < holdout.close.index[0]


def test_choose_zscore_returns_a_grid_value_and_ties_prefer_lower_z():
    # Exact line: no trades, all pnl 0 => pick the smaller Z.
    rng = np.random.default_rng(0)
    n = 160
    idx = pd.bdate_range("2020-01-01", periods=n)
    log_b = np.cumsum(rng.normal(0, 0.01, size=n)) + 4.0
    log_a = 0.4 + 0.85 * log_b
    close = np.exp(pd.DataFrame({"AAA": log_a, "BBB": log_b}, index=idx))
    prices = PricePanel(open=close.copy(), close=close)
    cfg = BacktestConfig(formation_days=120, trading_days=40, zscore_window=10)
    chosen = choose_zscore(prices, (2.5, 2.0), cfg)
    assert chosen == 2.0


def test_choose_zscore_from_rows_does_not_need_another_backtest():
    from pair_trade_backtest.zscore_sweep import choose_zscore_from_rows

    rows = [(2.5, 1.0, 0.5, 10), (2.0, 1.0, 0.4, 12), (2.25, 0.5, 0.3, 8)]
    assert choose_zscore_from_rows(rows) == 2.0
