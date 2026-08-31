from __future__ import annotations

from pathlib import Path

from pair_trade_backtest.universe import (
    SAMPLE_END,
    SAMPLE_START,
    default_universe_path,
    download_sample,
)


def main() -> None:
    cache_dir = Path("data/cache")
    result = download_sample(cache_dir)
    loaded = list(result.prices.close.columns)
    print(f"Universe file: {default_universe_path()}")
    print(f"Sample: {SAMPLE_START} .. {SAMPLE_END} (Yahoo end exclusive of later days)")
    print(f"Loaded {len(loaded)} tickers into {cache_dir.resolve()}")
    if result.skipped:
        print("Skipped:")
        for ticker in result.skipped:
            reason = result.skip_reasons.get(ticker, "")
            extra = f" ({reason})" if reason else ""
            print(f"  {ticker}{extra}")
    else:
        print("Skipped: (none)")


if __name__ == "__main__":
    main()
