from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pair_trade_backtest import BacktestConfig, PricePanel, run_backtest

ZSCORE_GRID = (2.0, 2.25, 2.5)


def split_walk_forward(
    prices: PricePanel,
    formation_days: int,
    trading_days: int,
    n_select_cycles: int = 2,
) -> tuple[PricePanel, PricePanel]:
    # Same 12/6 blocks as run_backtest (i += formation + trading).
    cycle = formation_days + trading_days
    cut = n_select_cycles * cycle
    if cut <= 0 or cut >= len(prices.close.index):
        raise ValueError("not enough sessions to split selection and holdout")
    if len(prices.close.index) - cut < cycle:
        raise ValueError("holdout is shorter than one formation/trading cycle")
    return (
        PricePanel(open=prices.open.iloc[:cut], close=prices.close.iloc[:cut]),
        PricePanel(open=prices.open.iloc[cut:], close=prices.close.iloc[cut:]),
    )


def choose_zscore_from_rows(rows: Sequence[tuple[float, float, float, int]]) -> float:
    # Highest selection total_pnl; ties keep the lower Z (closer to textbook 2.0).
    best_z = min(row[0] for row in rows)
    best_pnl = float("-inf")
    for z, pnl, _hit, _n in sorted(rows, key=lambda row: row[0]):
        if pnl > best_pnl:
            best_pnl = pnl
            best_z = z
    return float(best_z)


def choose_zscore(
    select_panel: PricePanel,
    grid: Sequence[float],
    base_config: BacktestConfig,
) -> float:
    return choose_zscore_from_rows(selection_rows(select_panel, grid, base_config))


def selection_rows(
    select_panel: PricePanel,
    grid: Sequence[float],
    base_config: BacktestConfig,
) -> list[tuple[float, float, float, int]]:
    rows: list[tuple[float, float, float, int]] = []
    for z in grid:
        cfg = base_config.model_copy(update={"zscore_entry": z})
        result = run_backtest(select_panel, cfg)
        rows.append((z, result.total_pnl, result.hit_rate, len(result.trades)))
    return rows


def main() -> None:
    import pandas as pd

    from pair_trade_backtest.prices import load_price_panel
    from pair_trade_backtest.universe import (
        SAMPLE_END,
        SAMPLE_START,
        load_frozen_universe,
    )

    def _no_network(ticker: str, start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame()

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
    print("Selection (cycles 1–2)")
    print("zscore  total_pnl  hit_rate  n_trades")
    rows = selection_rows(select, ZSCORE_GRID, cfg)
    for z, pnl, hit, n in rows:
        print(f"{z:6.2f}  {pnl:9.4f}  {hit:8.3f}  {n:8d}")
    chosen = choose_zscore_from_rows(rows)
    print(f"Chosen Z (max selection pnl, ties -> lower Z): {chosen}")
    frozen = cfg.model_copy(update={"zscore_entry": chosen})
    hold = run_backtest(holdout, frozen)
    print("Holdout (cycle 3 only, frozen Z)")
    print(
        f"trades {len(hold.trades)}  pnl {hold.total_pnl}  hit {hold.hit_rate}"
    )
    print("Holdout peek (not used to choose Z)")
    for z, pnl, hit, n in selection_rows(holdout, ZSCORE_GRID, cfg):
        print(f"{z:6.2f}  {pnl:9.4f}  {hit:8.3f}  {n:8d}")


if __name__ == "__main__":
    main()
