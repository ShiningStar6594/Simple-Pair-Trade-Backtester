# Pair Trade Backtesting System

A research-oriented Python backtester for statistical equity Pairs in the US
Banks and Capital Markets universe. The project focuses on realistic evaluation:
strict Formation and Trading Period separation, next-session-open execution,
transaction costs, and walk-forward parameter selection.

## Research Question

Can equity Pairs selected through correlation and Engle-Granger Cointegration
produce robust mean-reversion returns after realistic execution and holding
costs?

The goal is not to optimize a visually attractive backtest. It is to evaluate
the strategy without using future prices during Pair selection or parameter
choice.

## Methodology

Each walk-forward cycle contains:

1. **Formation Period — 252 sessions**
   - Screen every unordered Pair using absolute correlation of daily log returns.
   - Test surviving Pairs for Engle-Granger Cointegration.
   - Estimate and freeze the OLS Hedge Ratio and intercept.
2. **Trading Period — 126 sessions**
   - Apply the frozen relationship to Trading Period log Prices.
   - Compute a rolling 60-session Spread Z-Score.
   - Enter when the absolute Z-Score reaches the selected threshold.
   - Exit on a zero crossing, failed rolling ADF test, or period end.
3. **Execution and costs**
   - Generate signals from closing prices and Fill at the next session’s open.
   - Apply Commission and Bid-Ask Cost on entry and exit.
   - Accrue Borrow Cost on the short leg.

Z-Score candidates are compared using the first two cycles. The selected value
is then frozen before evaluation on the third-cycle holdout.

## Key Engineering Decisions

- A date-loop engine makes signal timing and next-open Fills explicit.
- Formation estimates remain frozen throughout each Trading Period.
- Adjusted closing prices drive statistical calculations; opening prices drive
  Fills and PnL.
- Each Pair allows at most one open position.
- Price downloads are cached by sample window and unavailable tickers are
  skipped with a recorded reason.
- Tests use deterministic synthetic Price fixtures and never depend on Yahoo.
- Review reports read from the local cache and do not silently fetch new data.

## Outputs

The review layer produces:

- holdout and selection equity curves;
- Spread and Z-Score charts with entry and exit Fill markers;
- after-cost total PnL, hit rate, mean and median trade PnL;
- realized-PnL Sharpe ratio and maximum drawdown;
- exit-reason proportions; and
- best and worst holdout Pairs ranked by PnL.

Generated research artifacts are written under `reports/`. Raw Yahoo Price files
remain local under `data/cache/`.

## Validation

The test suite covers:

- admission of synthetic cointegrated Pairs and rejection of unrelated series;
- absence of lookahead between Formation and Trading Periods;
- next-session-open entry and exit Fills;
- Z-Score, Statistical, and period-end exits;
- Commission, Bid-Ask Cost, Borrow Cost, and hand-checked PnL;
- flat Spread, invalid Fill price, and constant-window edge cases;
- cache isolation, missing data, and rate-limit handling; and
- walk-forward splitting, threshold selection, metrics, and chart generation.

## Limitations

- Daily bars do not model intraday execution, partial fills, market impact, or
  borrow availability.
- Results use a frozen sector universe rather than point-in-time index
  constituents.
- The reported equity curve aggregates realized trade PnL and is not a
  mark-to-market portfolio NAV.
- Pair notionals are evaluated independently; the system does not allocate
  capital or enforce portfolio-level exposure constraints.
- Statistical relationships can break after formation, so a passing
  Cointegration test does not guarantee profitable mean reversion.

## Project Structure

```text
pair_trade_backtest/   Backtest engine, Price loader, sweep, metrics, and charts
tests/                 Deterministic behavioral tests
data/                  Frozen Universe definition
reports/               Locally generated metrics and figures
docs/adr/              Design decisions
```
