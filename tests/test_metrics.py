"""Metrics seam: Trade blotter in, Sharpe / drawdown / table stats out. No charts."""

from __future__ import annotations

import pandas as pd
import pytest

from pair_trade_backtest import Trade
from pair_trade_backtest.metrics import equity_curve, pair_rows, summarize_window


def _trade(
    *,
    ticker_a: str = "AAA",
    ticker_b: str = "BBB",
    fill: str = "2020-01-02",
    exit_date: str = "2020-01-03",
    pnl: float,
    exit_reason: str = "zscore_target",
) -> Trade:
    return Trade(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        fill_date=pd.Timestamp(fill),
        fill_open_a=10.0,
        fill_open_b=20.0,
        side="short_spread",
        exit_date=pd.Timestamp(exit_date),
        exit_open_a=10.0,
        exit_open_b=20.0,
        exit_reason=exit_reason,
        hedge_ratio=1.0,
        friction_cost=0.0,
        borrow_cost=0.0,
        pnl=pnl,
    )


def test_equity_curve_is_cumulative_realized_pnl_on_exit_dates():
    trades = [
        _trade(exit_date="2020-01-02", pnl=0.10),
        _trade(exit_date="2020-01-03", pnl=-0.05),
    ]
    curve = equity_curve(trades)
    assert list(curve.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    assert curve.iloc[0] == pytest.approx(0.10)
    assert curve.iloc[1] == pytest.approx(0.05)


def test_same_exit_date_pnls_are_summed_on_the_curve():
    trades = [
        _trade(ticker_a="AAA", exit_date="2020-01-02", pnl=0.03),
        _trade(ticker_a="CCC", ticker_b="DDD", exit_date="2020-01-02", pnl=0.02),
    ]
    curve = equity_curve(trades)
    assert len(curve) == 1
    assert curve.iloc[0] == pytest.approx(0.05)


def test_summarize_window_uses_worked_sharpe_and_drawdown_literals():
    # Two exits: daily realized PnL 0.10 then -0.05.
    # mean = 0.025; sample std = sqrt(0.01125); Sharpe = mean/std * sqrt(252).
    trades = [
        _trade(exit_date="2020-01-02", pnl=0.10, exit_reason="zscore_target"),
        _trade(exit_date="2020-01-03", pnl=-0.05, exit_reason="period_end"),
    ]
    stats = summarize_window(trades, n_admitted_pairs=3)
    assert stats.n_admitted_pairs == 3
    assert stats.n_trades == 2
    assert stats.total_pnl == pytest.approx(0.05)
    assert stats.hit_rate == pytest.approx(0.5)
    assert stats.mean_trade_pnl == pytest.approx(0.025)
    assert stats.median_trade_pnl == pytest.approx(0.025)
    assert stats.share_zscore_target == pytest.approx(0.5)
    assert stats.share_statistical_exit == pytest.approx(0.0)
    assert stats.share_period_end == pytest.approx(0.5)
    assert stats.sharpe == pytest.approx(3.7416573867739413)
    assert stats.max_drawdown == pytest.approx(0.05)


def test_empty_blotter_has_zero_sharpe_and_drawdown():
    stats = summarize_window([], n_admitted_pairs=0)
    assert stats.n_trades == 0
    assert stats.total_pnl == 0.0
    assert stats.hit_rate == 0.0
    assert stats.sharpe == 0.0
    assert stats.max_drawdown == 0.0
    assert equity_curve([]).empty


def test_pair_rows_caps_at_top_and_bottom_by_pnl():
    trades = []
    for i in range(20):
        name = f"T{i:02d}"
        trades.append(
            _trade(
                ticker_a=name,
                ticker_b="ZZZ",
                exit_date="2020-01-02",
                pnl=float(i),
            )
        )
    rows = pair_rows(trades, cap=15)
    assert len(rows) == 15
    assert rows[0].ticker_a == "T19" and rows[0].total_pnl == pytest.approx(19.0)
    assert rows[-1].ticker_a == "T00" and rows[-1].total_pnl == pytest.approx(0.0)


def test_pick_review_pair_is_the_worst_total_pnl():
    from pair_trade_backtest.metrics import pick_review_pair

    trades = [
        _trade(ticker_a="WIN", ticker_b="ZZZ", pnl=1.0),
        _trade(ticker_a="LOSE", ticker_b="ZZZ", pnl=-2.0),
    ]
    assert pick_review_pair(trades) == ("LOSE", "ZZZ")


def test_format_window_block_prints_worked_sharpe():
    from pair_trade_backtest.metrics import format_window_block

    trades = [
        _trade(exit_date="2020-01-02", pnl=0.10, exit_reason="zscore_target"),
        _trade(exit_date="2020-01-03", pnl=-0.05, exit_reason="period_end"),
    ]
    text = format_window_block("Holdout (cycle 3, frozen Z=2.0) - headline", summarize_window(trades))
    assert "Holdout (cycle 3, frozen Z=2.0) - headline" in text
    assert "sharpe 3.742" in text
    assert "max_drawdown 0.0500" in text

