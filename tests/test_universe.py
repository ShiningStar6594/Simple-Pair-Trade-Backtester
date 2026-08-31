"""Frozen Universe + sample window. No live yfinance."""

from __future__ import annotations

import pandas as pd

from pair_trade_backtest.universe import (
    SAMPLE_END,
    SAMPLE_START,
    download_sample,
    load_frozen_universe,
)


def test_load_frozen_universe_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "universe.txt"
    path.write_text("# freeze as-of 2021-01\n\nJPM\n# note\nGS\n", encoding="utf-8")
    assert load_frozen_universe(path) == ["JPM", "GS"]


def test_shipped_universe_is_banks_cm_sized_and_not_spx():
    tickers = load_frozen_universe()
    assert 40 <= len(tickers) <= 50
    assert "JPM" in tickers and "GS" in tickers
    assert "AAPL" not in tickers
    assert "SIVB" not in tickers
    assert "WAL" in tickers and "SNV" in tickers and "IVZ" in tickers


def test_download_sample_uses_2021_2025_window_and_reports_skipped(tmp_path):
    uni = tmp_path / "universe.txt"
    uni.write_text("AAA\nDEAD\n", encoding="utf-8")
    idx = pd.bdate_range("2021-01-04", "2025-12-31")
    n = len(idx)
    bars = pd.DataFrame(
        {"Open": [10.0] * n, "Close": [10.2] * n},
        index=idx,
    )
    seen: list[tuple[str, str, str]] = []

    def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
        seen.append((ticker, start, end))
        if ticker == "DEAD":
            return pd.DataFrame()
        return bars

    result = download_sample(tmp_path / "cache", universe_path=uni, fetch=fetch)

    assert SAMPLE_START == "2021-01-01"
    assert SAMPLE_END == "2026-01-01"
    assert seen == [("AAA", SAMPLE_START, SAMPLE_END), ("DEAD", SAMPLE_START, SAMPLE_END)]
    assert result.skipped == ["DEAD"]
    assert result.skip_reasons["DEAD"] == "no_data"
    assert list(result.prices.close.columns) == ["AAA"]
