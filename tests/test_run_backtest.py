"""run_backtest seam: fake prices in, check spec behavior. No yfinance / no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pair_trade_backtest import BacktestConfig, PricePanel, run_backtest


def _business_days(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    # Weekdays only, like a stock calendar (no Sat/Sun)
    return pd.bdate_range(start, periods=n)


def _panel_from_log_close(log_close: pd.DataFrame) -> PricePanel:
    close = np.exp(log_close)  # tests build log Price; exp -> dollar close
    # Opens unused for admission tests; copy close so the panel has both columns
    return PricePanel(open=close.copy(), close=close)


def test_synthetic_cointegrated_pair_is_admitted():
    # BAC ≈ intercept + 0.85 * JPM + tiny noise => should pass corr + coint
    rng = np.random.default_rng(0)
    dates = _business_days(200)
    log_jpm = np.cumsum(rng.normal(0, 0.01, size=len(dates))) + 4.0  # random walk
    log_bac = 0.4 + 0.85 * log_jpm + rng.normal(0, 0.002, size=len(dates))
    log_close = pd.DataFrame({"BAC": log_bac, "JPM": log_jpm}, index=dates)
    prices = _panel_from_log_close(log_close)

    result = run_backtest(
        prices,
        BacktestConfig(formation_days=120, trading_days=40),  # shorter than 252/126 for a fast test
    )

    assert result.admitted_pairs == [("BAC", "JPM")]  # alphabetical A, B


def test_uncorrelated_noise_pair_is_not_admitted():
    # A wanders; B is just noise => not a Pair
    rng = np.random.default_rng(1)
    dates = _business_days(200)
    log_a = np.cumsum(rng.normal(0, 0.01, size=len(dates))) + 4.0
    log_b = rng.normal(0, 0.01, size=len(dates)) + 3.0
    prices = _panel_from_log_close(
        pd.DataFrame({"AAA": log_a, "BBB": log_b}, index=dates)
    )

    result = run_backtest(
        prices,
        BacktestConfig(formation_days=120, trading_days=40),
    )

    assert result.admitted_pairs == []


def test_pair_cointegrated_only_in_trading_period_is_not_admitted():
    # If they only line up in the Trading Period, formation must still reject (no lookahead)
    rng = np.random.default_rng(2)
    dates = _business_days(200)
    formation_days, trading_days = 120, 40
    log_b = np.cumsum(rng.normal(0, 0.01, size=len(dates))) + 4.0
    log_a = np.cumsum(rng.normal(0, 0.01, size=len(dates))) + 3.5
    # Only the Trading Period is a tight residual of B; formation stays independent.
    log_a[formation_days : formation_days + trading_days] = (
        0.3
        + 0.9 * log_b[formation_days : formation_days + trading_days]
        + rng.normal(0, 0.001, size=trading_days)
    )
    prices = _panel_from_log_close(
        pd.DataFrame({"AAA": log_a, "BBB": log_b}, index=dates)
    )

    result = run_backtest(
        prices,
        BacktestConfig(formation_days=formation_days, trading_days=trading_days),
    )

    assert result.admitted_pairs == []


def test_zscore_signal_fills_at_next_open_not_signal_close():
    # Shock close so |Z|>=2; Fill must be NEXT open (111/222), not that close
    rng = np.random.default_rng(3)
    formation_days, trading_days = 80, 25
    z_window = 10
    dates = _business_days(formation_days + trading_days)
    log_b = np.cumsum(rng.normal(0, 0.01, size=len(dates))) + 4.0
    log_a = 0.4 + 0.85 * log_b + rng.normal(0, 0.001, size=len(dates))
    shock_i = formation_days + z_window - 1
    log_a[shock_i] += 0.2  # blow out Spread on this close

    close = np.exp(pd.DataFrame({"AAA": log_a, "BBB": log_b}, index=dates))
    open_px = close.copy()
    fill_date = dates[shock_i + 1]  # tomorrow
    open_px.loc[fill_date, "AAA"] = 111.0  # planted Fill, not equal to yesterday close
    open_px.loc[fill_date, "BBB"] = 222.0
    prices = PricePanel(open=open_px, close=close)

    result = run_backtest(
        prices,
        BacktestConfig(
            formation_days=formation_days,
            trading_days=trading_days,
            zscore_window=z_window,
        ),
    )

    assert result.admitted_pairs == [("AAA", "BBB")]
    assert len(result.trades) >= 1
    trade = result.trades[0]
    assert trade.fill_date == fill_date
    assert trade.fill_open_a == 111.0
    assert trade.fill_open_b == 222.0
    assert trade.fill_open_a != float(close.loc[dates[shock_i], "AAA"])


def test_position_flattens_at_next_open_when_zscore_crosses_zero():
    # Enter on a shock, then put A back on the hedge line => Z crosses 0 => flatten next open
    rng = np.random.default_rng(4)
    formation_days, trading_days = 80, 30
    z_window = 10
    dates = _business_days(formation_days + trading_days)
    log_b = np.cumsum(rng.normal(0, 0.01, size=len(dates))) + 4.0
    log_a = 0.4 + 0.85 * log_b + rng.normal(0, 0.001, size=len(dates))
    shock_i = formation_days + z_window - 1
    revert_i = shock_i + 3
    log_a[shock_i:revert_i] += 0.2
    log_a[revert_i:] = 0.4 + 0.85 * log_b[revert_i:]

    close = np.exp(pd.DataFrame({"AAA": log_a, "BBB": log_b}, index=dates))
    open_px = close.copy()
    entry_date = dates[shock_i + 1]
    exit_date = dates[revert_i + 1]
    open_px.loc[entry_date, "AAA"] = 111.0
    open_px.loc[entry_date, "BBB"] = 222.0
    open_px.loc[exit_date, "AAA"] = 101.0
    open_px.loc[exit_date, "BBB"] = 202.0
    prices = PricePanel(open=open_px, close=close)

    result = run_backtest(
        prices,
        BacktestConfig(
            formation_days=formation_days,
            trading_days=trading_days,
            zscore_window=z_window,
        ),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.fill_date == entry_date
    assert trade.fill_open_a == 111.0
    assert trade.exit_date == exit_date
    assert trade.exit_open_a == 101.0
    assert trade.exit_open_b == 202.0
    assert trade.exit_reason == "zscore_target"


def test_statistical_exit_flattens_when_spread_adf_stops_rejecting():
    # After entry, Spread becomes a random walk (stays away from 0) => ADF p rises => statistical_exit
    rng = np.random.default_rng(5)
    formation_days, trading_days = 80, 40
    z_window, adf_window = 10, 12
    dates = _business_days(formation_days + trading_days)
    log_b = np.cumsum(rng.normal(0, 0.01, size=len(dates))) + 4.0
    log_a = 0.4 + 0.85 * log_b + rng.normal(0, 0.001, size=len(dates))
    shock_i = formation_days + z_window - 1
    log_a[shock_i] += 0.25
    after = shock_i + 1
    walk = 0.25 + np.cumsum(np.abs(rng.normal(0.01, 0.003, size=len(dates) - after)))
    log_a[after:] = 0.4 + 0.85 * log_b[after:] + walk

    close = np.exp(pd.DataFrame({"AAA": log_a, "BBB": log_b}, index=dates))
    open_px = close.copy()
    entry_date = dates[shock_i + 1]
    open_px.loc[entry_date, "AAA"] = 111.0
    open_px.loc[entry_date, "BBB"] = 222.0
    open_px.loc[dates[shock_i + 2] :, "AAA"] = 101.0  # whatever day ADF fires, Fill is this $
    open_px.loc[dates[shock_i + 2] :, "BBB"] = 202.0
    prices = PricePanel(open=open_px, close=close)

    result = run_backtest(
        prices,
        BacktestConfig(
            formation_days=formation_days,
            trading_days=trading_days,
            zscore_window=z_window,
            adf_window=adf_window,
        ),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.fill_date == entry_date
    assert trade.exit_reason == "statistical_exit"
    assert trade.exit_date is not None and trade.exit_date > trade.fill_date
    assert trade.exit_open_a == 101.0
    assert trade.exit_open_b == 202.0


def test_open_position_flattens_at_period_end():
    # Stay shocked so Z never comes back; short window so ADF never runs; flatten last open
    rng = np.random.default_rng(6)
    formation_days, trading_days = 80, 16
    z_window = 10
    dates = _business_days(formation_days + trading_days)
    log_b = np.cumsum(rng.normal(0, 0.01, size=len(dates))) + 4.0
    log_a = 0.4 + 0.85 * log_b + rng.normal(0, 0.001, size=len(dates))
    shock_i = formation_days + z_window - 1
    log_a[shock_i:] += 0.2

    close = np.exp(pd.DataFrame({"AAA": log_a, "BBB": log_b}, index=dates))
    open_px = close.copy()
    last_date = dates[-1]
    open_px.loc[last_date, "AAA"] = 99.0
    open_px.loc[last_date, "BBB"] = 199.0
    prices = PricePanel(open=open_px, close=close)

    result = run_backtest(
        prices,
        BacktestConfig(
            formation_days=formation_days,
            trading_days=trading_days,
            zscore_window=z_window,
            adf_window=60,
        ),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "period_end"
    assert trade.exit_date == last_date
    assert trade.exit_open_a == 99.0
    assert trade.exit_open_b == 199.0


def test_commission_and_bid_ask_charged_on_enter_and_exit():
    # Operational costs: broker fee + bid-ask haircut, both legs, enter AND exit (not borrow).
    rng = np.random.default_rng(4)
    formation_days, trading_days = 80, 30
    z_window = 10
    dates = _business_days(formation_days + trading_days)
    log_b = np.cumsum(rng.normal(0, 0.01, size=len(dates))) + 4.0
    log_a = 0.4 + 0.85 * log_b + rng.normal(0, 0.001, size=len(dates))
    shock_i = formation_days + z_window - 1
    revert_i = shock_i + 3
    log_a[shock_i:revert_i] += 0.2
    log_a[revert_i:] = 0.4 + 0.85 * log_b[revert_i:]
    prices = _panel_from_log_close(
        pd.DataFrame({"AAA": log_a, "BBB": log_b}, index=dates)
    )
    config = BacktestConfig(
        formation_days=formation_days,
        trading_days=trading_days,
        zscore_window=z_window,
        commission_bps=1.0,
        bid_ask_bps=5.0,
    )

    result = run_backtest(prices, config)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "zscore_target"
    bps = (config.commission_bps + config.bid_ask_bps) / 10_000
    # worked example: 2 turnovers * 6bp * ($1 A + $|hedge| B)
    expected = 2 * bps * (1.0 + abs(trade.hedge_ratio))
    assert trade.friction_cost == pytest.approx(expected)


def test_borrow_cost_accrues_on_short_leg_while_held():
    # 1%/year on the short notional, calendar days / 365. short_spread => short $1 of A.
    rng = np.random.default_rng(4)
    formation_days, trading_days = 80, 30
    z_window = 10
    dates = _business_days(formation_days + trading_days)
    log_b = np.cumsum(rng.normal(0, 0.01, size=len(dates))) + 4.0
    log_a = 0.4 + 0.85 * log_b + rng.normal(0, 0.001, size=len(dates))
    shock_i = formation_days + z_window - 1
    revert_i = shock_i + 3
    log_a[shock_i:revert_i] += 0.2
    log_a[revert_i:] = 0.4 + 0.85 * log_b[revert_i:]
    prices = _panel_from_log_close(
        pd.DataFrame({"AAA": log_a, "BBB": log_b}, index=dates)
    )

    result = run_backtest(
        prices,
        BacktestConfig(
            formation_days=formation_days,
            trading_days=trading_days,
            zscore_window=z_window,
            borrow_annual=0.01,
        ),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "short_spread"
    assert trade.exit_date is not None
    holding_years = (trade.exit_date - trade.fill_date).days / 365.0
    expected = 0.01 * 1.0 * holding_years  # short $1 of A
    assert trade.borrow_cost == pytest.approx(expected)


def test_trade_pnl_uses_opens_minus_friction_and_borrow():
    # $1 of A, $|hedge| of B; short_spread: profit if A falls / B rises. Net = gross - costs.
    rng = np.random.default_rng(4)
    formation_days, trading_days = 80, 30
    z_window = 10
    dates = _business_days(formation_days + trading_days)
    log_b = np.cumsum(rng.normal(0, 0.01, size=len(dates))) + 4.0
    log_a = 0.4 + 0.85 * log_b + rng.normal(0, 0.001, size=len(dates))
    shock_i = formation_days + z_window - 1
    revert_i = shock_i + 3
    log_a[shock_i:revert_i] += 0.2
    log_a[revert_i:] = 0.4 + 0.85 * log_b[revert_i:]
    close = np.exp(pd.DataFrame({"AAA": log_a, "BBB": log_b}, index=dates))
    open_px = close.copy()
    entry_date = dates[shock_i + 1]
    exit_date = dates[revert_i + 1]
    open_px.loc[entry_date, "AAA"] = 111.0
    open_px.loc[entry_date, "BBB"] = 222.0
    open_px.loc[exit_date, "AAA"] = 101.0
    open_px.loc[exit_date, "BBB"] = 202.0
    prices = PricePanel(open=open_px, close=close)

    result = run_backtest(
        prices,
        BacktestConfig(
            formation_days=formation_days,
            trading_days=trading_days,
            zscore_window=z_window,
        ),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "short_spread"
    h = abs(trade.hedge_ratio)
    # short $1 of A at 111, cover at 101; long $h of B at 222, sell at 202
    gross = (1.0 / 111.0) * (111.0 - 101.0) + (h / 222.0) * (202.0 - 222.0)
    expected = gross - trade.friction_cost - trade.borrow_cost
    assert trade.pnl == pytest.approx(expected)
    assert result.total_pnl == pytest.approx(expected)
    assert result.hit_rate == pytest.approx(1.0 if expected > 0 else 0.0)
    # Independent costs (not trade.friction_cost / borrow_cost), so a matching bug cannot hide.
    bps = (1.0 + 5.0) / 10_000
    friction_literal = 2 * bps * (1.0 + h)
    borrow_literal = 0.01 * 1.0 * (trade.exit_date - trade.fill_date).days / 365.0
    assert trade.friction_cost == pytest.approx(friction_literal)
    assert trade.borrow_cost == pytest.approx(borrow_literal)
    assert trade.pnl == pytest.approx(gross - friction_literal - borrow_literal)
    assert trade.fill_open_a != float(close.loc[trade.fill_date, "AAA"])


def test_flat_z_does_not_open_a_trade():
    # Exact hedge line: residual is 0, Z is not finite / not |Z|>=2 => wait, no blotter row.
    # (Tiny noise + a 10-day Z window can still print |Z|>=2; that is not an engine bug.)
    rng = np.random.default_rng(7)
    formation_days, trading_days = 80, 25
    z_window = 10
    dates = _business_days(formation_days + trading_days)
    log_b = np.cumsum(rng.normal(0, 0.01, size=len(dates))) + 4.0
    log_a = 0.4 + 0.85 * log_b
    prices = _panel_from_log_close(pd.DataFrame({"AAA": log_a, "BBB": log_b}, index=dates))

    result = run_backtest(
        prices,
        BacktestConfig(
            formation_days=formation_days,
            trading_days=trading_days,
            zscore_window=z_window,
        ),
    )

    assert result.admitted_pairs == [("AAA", "BBB")]
    assert result.trades == []


def test_zero_fill_open_does_not_crash():
    rng = np.random.default_rng(4)
    formation_days, trading_days = 80, 25
    z_window = 10
    dates = _business_days(formation_days + trading_days)
    log_b = np.cumsum(rng.normal(0, 0.01, size=len(dates))) + 4.0
    log_a = 0.4 + 0.85 * log_b + rng.normal(0, 0.001, size=len(dates))
    shock_i = formation_days + z_window - 1
    log_a[shock_i] += 0.2
    close = np.exp(pd.DataFrame({"AAA": log_a, "BBB": log_b}, index=dates))
    open_px = close.copy()
    open_px.loc[dates[shock_i + 1], "AAA"] = 0.0
    prices = PricePanel(open=open_px, close=close)

    result = run_backtest(
        prices,
        BacktestConfig(
            formation_days=formation_days,
            trading_days=trading_days,
            zscore_window=z_window,
        ),
    )

    assert all(t.fill_open_a > 0 and t.fill_open_b > 0 for t in result.trades)


def test_constant_spread_after_entry_does_not_crash_adf():
    rng = np.random.default_rng(8)
    formation_days, trading_days = 80, 40
    z_window, adf_window = 10, 12
    dates = _business_days(formation_days + trading_days)
    log_b = np.cumsum(rng.normal(0, 0.01, size=len(dates))) + 4.0
    log_a = 0.4 + 0.85 * log_b + rng.normal(0, 0.001, size=len(dates))
    shock_i = formation_days + z_window - 1
    log_a[shock_i] += 0.2
    # After entry, freeze A on the exact hedge line => constant residual => adfuller used to raise.
    log_a[shock_i + 1 :] = 0.4 + 0.85 * log_b[shock_i + 1 :]
    close = np.exp(pd.DataFrame({"AAA": log_a, "BBB": log_b}, index=dates))
    prices = PricePanel(open=close.copy(), close=close)

    result = run_backtest(
        prices,
        BacktestConfig(
            formation_days=formation_days,
            trading_days=trading_days,
            zscore_window=z_window,
            adf_window=adf_window,
        ),
    )

    assert isinstance(result.trades, list)



