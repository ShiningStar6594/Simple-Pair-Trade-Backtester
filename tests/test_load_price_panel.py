"""load_price_panel seam: injected fetch + disk cache. No live yfinance."""

from __future__ import annotations

import pandas as pd
import pytest

from pair_trade_backtest.prices import load_price_panel


def _window_bars(first: str, last: str) -> pd.DataFrame:
    idx = pd.bdate_range(first, last)
    n = len(idx)
    return pd.DataFrame(
        {"Open": [10.0 + i * 0.1 for i in range(n)], "Close": [10.2 + i * 0.1 for i in range(n)]},
        index=idx,
    )


def _bars() -> pd.DataFrame:
    # Must cover start=2017-01-01, end=2017-01-10 (last weekday before end is 01-09).
    return _window_bars("2017-01-03", "2017-01-09")


def test_load_price_panel_writes_cache_and_reuses_it(tmp_path):
    calls: list[str] = []

    def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
        calls.append(ticker)
        return _bars()

    first = load_price_panel(
        ["AAA"],
        tmp_path,
        start="2017-01-01",
        end="2017-01-10",
        fetch=fetch,
    )
    second = load_price_panel(
        ["AAA"],
        tmp_path,
        start="2017-01-01",
        end="2017-01-10",
        fetch=fetch,
    )

    assert calls == ["AAA"]
    assert (tmp_path / "2017-01-01_2017-01-10" / "AAA.csv").is_file()
    assert list(first.prices.close.columns) == ["AAA"]
    assert first.prices.close.loc[pd.Timestamp("2017-01-03"), "AAA"] == pytest.approx(10.2)
    assert first.prices.open.loc[pd.Timestamp("2017-01-03"), "AAA"] == pytest.approx(10.0)
    assert first.skipped == []
    pd.testing.assert_frame_equal(second.prices.close, first.prices.close, check_freq=False)


def test_missing_ticker_is_skipped_and_listed(tmp_path):
    def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
        if ticker == "DEAD":
            return pd.DataFrame()
        return _bars()

    result = load_price_panel(
        ["AAA", "DEAD"],
        tmp_path,
        start="2017-01-01",
        end="2017-01-10",
        fetch=fetch,
    )

    assert result.skipped == ["DEAD"]
    assert result.skip_reasons["DEAD"] == "no_data"
    assert list(result.prices.close.columns) == ["AAA"]
    assert not (tmp_path / "2017-01-01_2017-01-10" / "DEAD.csv").exists()


def test_incomplete_history_is_skipped_not_on_panel(tmp_path):
    def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
        if ticker == "SIVB":
            return _window_bars("2021-01-04", "2023-03-10")
        return _window_bars("2021-01-04", "2025-12-31")

    result = load_price_panel(
        ["JPM", "SIVB"],
        tmp_path,
        start="2021-01-01",
        end="2026-01-01",
        fetch=fetch,
    )

    assert result.skipped == ["SIVB"]
    assert result.skip_reasons["SIVB"] == "incomplete"
    assert list(result.prices.close.columns) == ["JPM"]
    assert not result.prices.close.isna().any().any()


def test_two_complete_names_make_a_nan_free_panel(tmp_path):
    def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
        extra = 0.0 if ticker == "JPM" else 1.0
        bars = _window_bars("2021-01-04", "2025-12-31")
        return bars + extra

    result = load_price_panel(
        ["JPM", "GS"],
        tmp_path,
        start="2021-01-01",
        end="2026-01-01",
        fetch=fetch,
    )

    assert result.skipped == []
    assert list(result.prices.close.columns) == ["JPM", "GS"]
    assert not result.prices.close.isna().any().any()
    assert not result.prices.open.isna().any().any()


def test_yahoo_extra_ohlc_columns_are_not_cached_or_on_panel(tmp_path):
    # Yahoo sends High/Low/Volume/Adj Close; engine only needs Open + Close.
    idx = pd.bdate_range("2017-01-03", "2017-01-09")
    n = len(idx)

    def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": [10.0 + i * 0.1 for i in range(n)],
                "High": [99.0] * n,
                "Low": [1.0] * n,
                "Close": [10.2 + i * 0.1 for i in range(n)],
                "Adj Close": [50.0] * n,
                "Volume": [1_000_000] * n,
            },
            index=idx,
        )

    result = load_price_panel(
        ["AAA"],
        tmp_path,
        start="2017-01-01",
        end="2017-01-10",
        fetch=fetch,
    )
    cached = pd.read_csv(tmp_path / "2017-01-01_2017-01-10" / "AAA.csv")

    assert result.skipped == []
    assert list(result.prices.close.columns) == ["AAA"]
    assert list(result.prices.open.columns) == ["AAA"]
    assert "Volume" not in cached.columns
    assert "High" not in cached.columns
    assert "Adj Close" not in cached.columns
    assert set(c for c in cached.columns if c != cached.columns[0]) <= {"Open", "Close"} or set(
        cached.columns
    ) >= {"Open", "Close"}
    assert "Open" in cached.columns and "Close" in cached.columns
    assert "High" not in list(result.prices.close.columns)


def test_two_date_windows_do_not_share_cache_files(tmp_path):
    def fetch_short(ticker: str, start: str, end: str) -> pd.DataFrame:
        return _bars()

    def fetch_long(ticker: str, start: str, end: str) -> pd.DataFrame:
        return _window_bars("2021-01-04", "2025-12-31")

    load_price_panel(["AAA"], tmp_path, start="2017-01-01", end="2017-01-10", fetch=fetch_short)
    load_price_panel(["AAA"], tmp_path, start="2021-01-01", end="2026-01-01", fetch=fetch_long)

    short_path = tmp_path / "2017-01-01_2017-01-10" / "AAA.csv"
    long_path = tmp_path / "2021-01-01_2026-01-01" / "AAA.csv"
    assert short_path.is_file() and long_path.is_file()
    short_n = len(pd.read_csv(short_path, index_col=0, parse_dates=True))
    long_n = len(pd.read_csv(long_path, index_col=0, parse_dates=True))
    assert long_n > short_n


def test_rate_limit_then_success_is_not_skipped(tmp_path):
    from pair_trade_backtest.prices import FetchRateLimit

    calls: list[int] = []

    def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
        calls.append(1)
        if len(calls) == 1:
            raise FetchRateLimit("429")
        return _bars()

    result = load_price_panel(
        ["AAA"],
        tmp_path,
        start="2017-01-01",
        end="2017-01-10",
        fetch=fetch,
    )

    assert len(calls) == 2
    assert result.skipped == []
    assert list(result.prices.close.columns) == ["AAA"]


def test_persistent_rate_limit_is_skipped_with_reason(tmp_path):
    from pair_trade_backtest.prices import FetchRateLimit

    def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
        raise FetchRateLimit("429")

    result = load_price_panel(
        ["AAA"],
        tmp_path,
        start="2017-01-01",
        end="2017-01-10",
        fetch=fetch,
    )

    assert result.skipped == ["AAA"]
    assert result.skip_reasons["AAA"] == "rate_limit"
    assert not (tmp_path / "2017-01-01_2017-01-10" / "AAA.csv").exists()
