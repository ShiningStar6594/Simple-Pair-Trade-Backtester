from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pair_trade_backtest.prices import FetchFn, PriceLoadResult, load_price_panel

# 252+126=378 sessions/cycle. ~5 calendar years ≈ 3 cycles (not 4).
SAMPLE_START = "2021-01-01"
SAMPLE_END = "2026-01-01"
FREEZE_AS_OF = "2021-01"


def default_universe_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "universe_2021.txt"


def load_frozen_universe(path: Path | None = None) -> list[str]:
    # Checked-in list only. Comments and blanks are not tickers.
    source = Path(path) if path is not None else default_universe_path()
    tickers: list[str] = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tickers.append(line.split()[0].upper())
    return tickers


def download_sample(
    cache_dir: Path,
    universe_path: Path | None = None,
    tickers: Sequence[str] | None = None,
    *,
    fetch: FetchFn | None = None,
) -> PriceLoadResult:
    names = list(tickers) if tickers is not None else load_frozen_universe(universe_path)
    return load_price_panel(
        names,
        cache_dir,
        SAMPLE_START,
        SAMPLE_END,
        fetch=fetch,
    )
