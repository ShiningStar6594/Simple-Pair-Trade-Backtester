"""Charts write PNG files. No pixel snapshots."""

from __future__ import annotations

import pandas as pd

from pair_trade_backtest import Trade
from pair_trade_backtest.charts import save_equity_curve, save_pair_path


def test_save_equity_curve_writes_png(tmp_path):
    trades = [
        Trade(
            ticker_a="AAA",
            ticker_b="BBB",
            fill_date=pd.Timestamp("2020-01-02"),
            fill_open_a=10.0,
            fill_open_b=20.0,
            side="short_spread",
            exit_date=pd.Timestamp("2020-01-03"),
            exit_open_a=10.0,
            exit_open_b=20.0,
            exit_reason="zscore_target",
            pnl=0.10,
        )
    ]
    path = tmp_path / "equity.png"
    save_equity_curve(trades, path, "Holdout equity curve")
    assert path.is_file() and path.stat().st_size > 0


def test_save_pair_path_writes_png(tmp_path):
    idx = pd.bdate_range("2020-01-02", periods=5)
    spread = pd.Series([0.0, 0.1, 0.0, -0.1, 0.0], index=idx)
    zscore = pd.Series([0.0, 2.1, 0.5, -2.1, 0.0], index=idx)
    trades = [
        Trade(
            ticker_a="AAA",
            ticker_b="BBB",
            fill_date=idx[1],
            fill_open_a=10.0,
            fill_open_b=20.0,
            side="short_spread",
            exit_date=idx[2],
            exit_open_a=10.0,
            exit_open_b=20.0,
            exit_reason="zscore_target",
            pnl=0.01,
        )
    ]
    path = tmp_path / "pair.png"
    save_pair_path(spread, zscore, trades, path, "Holdout Pair AAA/BBB")
    assert path.is_file() and path.stat().st_size > 0


def test_figure_paths_lists_only_existing_pngs(tmp_path):
    from pair_trade_backtest.review import figure_paths

    (tmp_path / "equity_holdout.png").write_bytes(b"x")
    assert figure_paths(tmp_path) == [tmp_path / "equity_holdout.png"]

