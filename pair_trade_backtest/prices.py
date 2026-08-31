from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pair_trade_backtest import PricePanel

FetchFn = Callable[[str, str, str], pd.DataFrame]


class FetchRateLimit(Exception):
    """Yahoo (or injected fetch) asked us to back off. Not a delist."""


@dataclass(frozen=True)
class PriceLoadResult:
    prices: PricePanel
    skipped: list[str]
    skip_reasons: dict[str, str] = field(default_factory=dict)


def cache_window_dir(cache_dir: Path, start: str, end: str) -> Path:
    return Path(cache_dir) / f"{start}_{end}"


def load_price_panel(
    tickers: Sequence[str],
    cache_dir: Path,
    start: str,
    end: str,
    *,
    fetch: FetchFn | None = None,
) -> PriceLoadResult:
    # Disk cache first so a second run does not call yfinance.
    # fetch is injected in tests; default is Yahoo.
    cache_dir = Path(cache_dir)
    window = cache_window_dir(cache_dir, start, end)
    window.mkdir(parents=True, exist_ok=True)
    fetch_fn = fetch or _yf_fetch
    use_backoff = fetch is None
    opens: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    skipped: list[str] = []
    skip_reasons: dict[str, str] = {}

    for ticker in tickers:
        path = window / f"{ticker}.csv"
        if path.exists():
            df = _read_cached(path)
        else:
            raw, rate_failed = _fetch_with_retries(fetch_fn, ticker, start, end, use_backoff)
            if rate_failed:
                skipped.append(ticker)
                skip_reasons[ticker] = "rate_limit"
                continue
            df = _normalize_ohlc(raw)
            if df.empty:
                skipped.append(ticker)
                skip_reasons[ticker] = "no_data"
                continue
            df.to_csv(path)

        if not _covers_sample_window(df, start, end):
            skipped.append(ticker)
            skip_reasons[ticker] = "incomplete"
            continue
        opens[ticker] = df["Open"]
        closes[ticker] = df["Close"]

    if opens:
        open_df = pd.concat(opens, axis=1, join="inner")
        close_df = pd.concat(closes, axis=1, join="inner")
        open_df = open_df.dropna(how="any")
        close_df = close_df.reindex(open_df.index).dropna(how="any")
        open_df = open_df.reindex(close_df.index)
        open_df.index.name = None
        close_df.index.name = None
        open_df.columns.name = None
        close_df.columns.name = None
    else:
        open_df = pd.DataFrame()
        close_df = pd.DataFrame()

    return PriceLoadResult(
        prices=PricePanel(open=open_df, close=close_df),
        skipped=skipped,
        skip_reasons=skip_reasons,
    )


def _is_rate_limit(exc: BaseException) -> bool:
    if isinstance(exc, FetchRateLimit):
        return True
    name = type(exc).__name__
    if name == "YFRateLimitError":
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many" in text


def _fetch_with_retries(
    fetch_fn: FetchFn,
    ticker: str,
    start: str,
    end: str,
    use_backoff: bool,
) -> tuple[pd.DataFrame | None, bool]:
    last_rate = False
    for attempt in range(3):
        try:
            return fetch_fn(ticker, start, end), False
        except Exception as exc:
            if not _is_rate_limit(exc):
                return pd.DataFrame(), False
            last_rate = True
            if attempt < 2 and use_backoff:
                time.sleep(2 * (2**attempt))
            elif attempt < 2:
                continue
    return None, last_rate


def _covers_sample_window(df: pd.DataFrame, start: str, end: str) -> bool:
    # Full [start, end): first weekday near start, last weekday before end.
    # Delists that stop early (SIVB) fail last-bar. New Year listing lag is OK.
    if df.empty:
        return False
    last_day = pd.Timestamp(end) - pd.Timedelta(days=1)
    sessions = pd.bdate_range(start, last_day)
    if len(sessions) == 0:
        return False
    need_first = sessions[0]
    need_last = sessions[-1]
    first_bar = pd.Timestamp(df.index.min())
    last_bar = pd.Timestamp(df.index.max())
    if first_bar > need_first + pd.Timedelta(days=7):
        return False
    if last_bar < need_last:
        return False
    return not bool(df[["Open", "Close"]].isna().any().any())


def _read_cached(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _normalize_ohlc(df)


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "Close"])
    flat = df.copy()
    if isinstance(flat.columns, pd.MultiIndex):
        flat.columns = [str(c[0]) for c in flat.columns]
    cols = {str(c).strip().title(): c for c in flat.columns}
    if "Open" not in cols or "Close" not in cols:
        return pd.DataFrame(columns=["Open", "Close"])
    out = pd.DataFrame(
        {
            "Open": pd.to_numeric(flat[cols["Open"]], errors="coerce"),
            "Close": pd.to_numeric(flat[cols["Close"]], errors="coerce"),
        },
        index=pd.to_datetime(flat.index),
    )
    ix = pd.DatetimeIndex(pd.to_datetime(out.index))
    if ix.tz is not None:
        ix = ix.tz_localize(None)
    out.index = pd.DatetimeIndex(ix.values)
    out.index.name = None
    return out.dropna(how="any")


def _yf_fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError

    # auto_adjust=True: Close is split/dividend adjusted (our Price).
    try:
        return yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except YFRateLimitError as exc:
        raise FetchRateLimit(str(exc)) from exc
