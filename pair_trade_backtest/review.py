from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from pair_trade_backtest import BacktestConfig, PricePanel, _formation_fit, run_backtest
from pair_trade_backtest.charts import save_equity_curve, save_pair_path, spread_and_zscore
from pair_trade_backtest.metrics import (
    format_pair_table,
    format_window_block,
    pair_rows,
    pick_review_pair,
    summarize_result,
)
from pair_trade_backtest.zscore_sweep import split_walk_forward

REPORTS = Path("reports")
FROZEN_Z = 2.0


def _no_network(ticker: str, start: str, end: str) -> pd.DataFrame:
    return pd.DataFrame()


def _write_review(
    select: PricePanel,
    holdout: PricePanel,
    config: BacktestConfig,
    out_dir: Path,
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = config.model_copy(update={"zscore_entry": FROZEN_Z})
    select_result = run_backtest(select, frozen)
    hold_result = run_backtest(holdout, frozen)
    select_stats = summarize_result(select_result)
    hold_stats = summarize_result(hold_result)
    hold_pairs = pair_rows(hold_result.trades, cap=15)

    blocks = [
        format_window_block("Holdout (cycle 3, frozen Z=2.0) - headline", hold_stats),
        format_window_block("Selection (cycles 1-2, same frozen Z=2.0) - labeled, not the test", select_stats),
        "Holdout per-Pair (top/bottom by PnL, cap 15)",
        format_pair_table(hold_pairs),
    ]
    text = "\n\n".join(blocks) + "\n"
    (out_dir / "metrics.txt").write_text(text, encoding="utf-8")

    save_equity_curve(
        hold_result.trades,
        out_dir / "equity_holdout.png",
        "Holdout equity curve (cycle 3, frozen Z=2.0, after costs)",
    )
    save_equity_curve(
        select_result.trades,
        out_dir / "equity_selection.png",
        "Selection equity curve (cycles 1-2, frozen Z=2.0, after costs)",
    )

    chosen = pick_review_pair(hold_result.trades)
    if chosen is not None:
        ticker_a, ticker_b = chosen
        form = holdout.close.iloc[: frozen.formation_days]
        intercept, hedge = _formation_fit(form[ticker_a], form[ticker_b])
        trade_close_a = holdout.close[ticker_a].iloc[
            frozen.formation_days : frozen.formation_days + frozen.trading_days
        ]
        trade_close_b = holdout.close[ticker_b].iloc[
            frozen.formation_days : frozen.formation_days + frozen.trading_days
        ]
        spread, zscore = spread_and_zscore(
            trade_close_a,
            trade_close_b,
            intercept,
            hedge,
            frozen.zscore_window,
        )
        pair_trades = [
            t
            for t in hold_result.trades
            if t.ticker_a == ticker_a and t.ticker_b == ticker_b
        ]
        save_pair_path(
            spread,
            zscore,
            pair_trades,
            out_dir / "pair_path_holdout.png",
            (
                f"Holdout Pair {ticker_a}/{ticker_b} Spread and Z-Score "
                f"(blue=entry Fill, orange=exit Fill)"
            ),
            zscore_entry=FROZEN_Z,
        )
    return text


def figure_paths(out_dir: Path) -> list[Path]:
    names = ("equity_holdout.png", "equity_selection.png", "pair_path_holdout.png")
    return [path for name in names if (path := Path(out_dir) / name).is_file()]


def open_figures(paths: list[Path]) -> None:
    # Photos / default PNG app. A terminal cannot draw matplotlib.
    for path in paths:
        resolved = path.resolve()
        if sys.platform == "win32":
            os.startfile(resolved)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{resolved}"')
        else:
            os.system(f'xdg-open "{resolved}"')


def main() -> None:
    from pair_trade_backtest.prices import load_price_panel
    from pair_trade_backtest.universe import (
        SAMPLE_END,
        SAMPLE_START,
        load_frozen_universe,
    )

    loaded = load_price_panel(
        load_frozen_universe(),
        Path("data/cache"),
        SAMPLE_START,
        SAMPLE_END,
        fetch=_no_network,
    )
    if loaded.prices.close.empty:
        print("No cached bars for this sample window. Run: python -m pair_trade_backtest.download")
        return
    cfg = BacktestConfig()
    select, holdout = split_walk_forward(
        loaded.prices, cfg.formation_days, cfg.trading_days
    )
    text = _write_review(select, holdout, cfg, REPORTS)
    print(text, end="")
    paths = figure_paths(REPORTS)
    print(f"Wrote figures and metrics under {REPORTS.resolve()}")
    for path in paths:
        print(f"  {path.resolve()}")
    open_figures(paths)


if __name__ == "__main__":
    main()
