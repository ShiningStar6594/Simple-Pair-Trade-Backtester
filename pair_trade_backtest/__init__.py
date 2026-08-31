from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict
from statsmodels.tsa.stattools import adfuller, coint


class BacktestConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    formation_days: int = 252 #1 year deducting weekends and reserved some for holidays
    trading_days: int = 126 #1 year / 2 for 6 months period
    corr_threshold: float = 0.5 # Throw away unrelated stocks
    #spread = log(A) - intercept - hedge * log(B)
    #log(A) = intercept + Hedge Ratio × log(B) + error(spread)
    coint_pvalue: float = 0.05 # 
    zscore_entry: float = 2.0 #Spread at time t - mean / sd 
    #short A if >= 2, long A if <= -2
    zscore_window: int = 60 # re cal every day for 60 days period
    adf_window: int = 60 #did adf test for every 60 days
    commission_bps: float = 1.0 #fee to broker for each trade (per leg, enter AND exit)
    bid_ask_bps: float = 5.0  # worse fill vs mid: buy ask / sell bid (per leg, enter AND exit)
    borrow_annual: float = 0.01 #1% per year rent on the SHORT leg while the trade is open


@dataclass(frozen=True)
class PricePanel:
    open: pd.DataFrame #opening price of the stock
    close: pd.DataFrame #closing price of the stock


@dataclass
class Trade:
    ticker_a: str #stock A
    ticker_b: str #stock B
    fill_date: pd.Timestamp #date when the trade is filled and executed
    fill_open_a: float #opening price of stock A at fill date
    fill_open_b: float #opening price of stock B at fill date
    side: str #short_spread or long_spread
    exit_date: pd.Timestamp | None = None #date when the trade is exited and closed
    exit_open_a: float | None = None #closing price of stock A at exit date
    # price for trading 
    exit_open_b: float | None = None #closing price of stock B at exit date
    # logged and for series of the realtionship, which are built at the end of the day
    # so that the spread is calculated and the zscore is calculated
    exit_reason: str | None = None #reason for exit
    hedge_ratio: float = 0.0  # frozen OLS slope; we hold $1 of A and $|hedge| of B
    friction_cost: float = 0.0  # commission + bid-ask only; added on enter, added again on exit
    borrow_cost: float = 0.0  # 1%/year * short $ * calendar days / 365; set at exit
    pnl: float = 0.0  # share notionals × open moves, then minus friction and borrow


@dataclass
class BacktestResult:
    admitted_pairs: list[tuple[str, str]] = field(default_factory=list)
    #list of pairs that are admitted and allowed to trade
    trades: list[Trade] = field(default_factory=list)
    #list of trades that are executed and executed (trading info)
    total_pnl: float = 0.0  # sum of Trade.pnl (already after costs)
    hit_rate: float = 0.0  # fraction of trades with pnl > 0; 0.0 if none



def run_backtest(prices: PricePanel, config: BacktestConfig) -> BacktestResult:
    # Walk the calendar in 12-month form + 6-month trade cycles.
    # Formation Period: screen pairs, freeze intercept + hedge. No trades.
    # Trading Period: trade only admitted pairs; formation cannot see these prices.
    dates = prices.close.index
    formation_days = config.formation_days
    trading_days = config.trading_days
    tickers = sorted(str(c) for c in prices.close.columns)  # A = first ticker, B = second
    admitted: list[tuple[str, str]] = []  # pairs that passed corr + Engle-Granger
    trades: list[Trade] = []

    i = 0  # start index of the current cycle
    while i + formation_days + trading_days <= len(dates):
        form = prices.close.iloc[i : i + formation_days]  # closes used only to admit / fit
        trade_slice = slice(i + formation_days, i + formation_days + trading_days)
        for a_idx, ticker_a in enumerate(tickers):
            for ticker_b in tickers[a_idx + 1 :]:  # each unordered pair once
                # Cheap |corr| screen, then coint p-value < 0.05 on log prices
                if not _admits_pair(form[ticker_a], form[ticker_b], config):
                    continue
                pair = (ticker_a, ticker_b)
                if pair not in admitted:
                    admitted.append(pair)
                # OLS: log(A) = intercept + hedge * log(B) + spread  (frozen for trading)
                intercept, hedge = _formation_fit(form[ticker_a], form[ticker_b])
                # Date-loop: Z-Score signals, Fill next open, flatten on 0 / ADF / period end
                trades.extend(
                    _trades_for_pair(
                        ticker_a,
                        ticker_b,
                        prices,
                        trade_slice,
                        intercept,
                        hedge,
                        config,
                    )
                )
        i += formation_days + trading_days  # next cycle starts after this trade window

    total_pnl = sum(t.pnl for t in trades)
    hit_rate = (
        sum(1 for t in trades if t.pnl > 0) / len(trades) if trades else 0.0
    )
    return BacktestResult(
        admitted_pairs=admitted,
        trades=trades,
        total_pnl=total_pnl,
        hit_rate=hit_rate,
    )


def _admits_pair(close_a: pd.Series, close_b: pd.Series, config: BacktestConfig) -> bool:
    # Stats see log(close) only. Close is the Price series (relationship / signal).
    # Open is just the Fill later — a dollar price, not logged, not in coint.
    # log so hedge = elasticity ("if B moves 1%, A moves hedge%"), not raw dollars.
    log_a = np.log(close_a.to_numpy(dtype=float))
    log_b = np.log(close_b.to_numpy(dtype=float))
    # np.diff(log) ≈ daily returns. |corr| >= 0.5 = cheap "are they related?" filter
    corr = np.corrcoef(np.diff(log_a), np.diff(log_b))[0, 1]
    if not np.isfinite(corr) or abs(corr) < config.corr_threshold:
        return False
    # Engle-Granger: OLS residual then ADF. Small p => Spread not a random walk => admit
    # Dickey-Fuller table (MacKinnon p-values): coint's p is not a normal t-test
    _, pvalue, _ = coint(log_a, log_b)
    return bool(pvalue < config.coint_pvalue)


def _formation_fit(close_a: pd.Series, close_b: pd.Series) -> tuple[float, float]:
    # Same log(close) as coint. Frozen for trading; do not refit on trading days.
    # log(A) = intercept + hedge * log(B) + spread
    y = np.log(close_a.to_numpy(dtype=float))
    x = np.log(close_b.to_numpy(dtype=float))
    params = sm.OLS(y, sm.add_constant(x)).fit().params #Ordinary Least Squares
    
    #y = what we are trying to predict(A), x = what we are using to predict it (B)
    #params[0] = intercept, params[1] = Hedge Ratio
    return float(params[0]), float(params[1])  # intercept, Hedge Ratio


def _turnover_friction(config: BacktestConfig, hedge: float) -> float:
    # One Fill (enter OR exit), both legs.
    # 1 bp = 0.01% so divide bps by 10_000. Not borrow (that accrues while short).
    # $1 of A + $|hedge| of B. Commission paid to broker; bid-ask is the quote haircut.
    rate = (config.commission_bps + config.bid_ask_bps) / 10_000
    return rate * (1.0 + abs(hedge))


def _borrow_cost(
    config: BacktestConfig,
    side: str,
    hedge: float,
    fill_date: pd.Timestamp,
    exit_date: pd.Timestamp,
) -> float:
    # Rent the short: short_spread => short $1 of A; long_spread => short $|hedge| of B.
    # Calendar days (weekends count). Not a Fill fee.
    short_notional = 1.0 if side == "short_spread" else abs(hedge)
    holding_years = max((exit_date - fill_date).days, 0) / 365.0
    return config.borrow_annual * short_notional * holding_years


def _round_trip_pnl(trade: Trade) -> float:
    # Shares frozen at entry: $1 / fill_open_a of A, $|hedge| / fill_open_b of B.
    # short_spread: short A, long B. long_spread: long A, short B.
    # Uses Fill opens only — closes are for the signal, not dollars.
    h = abs(trade.hedge_ratio)
    fill_a = trade.fill_open_a
    fill_b = trade.fill_open_b
    exit_a = trade.exit_open_a
    exit_b = trade.exit_open_b
    if exit_a is None or exit_b is None:
        return 0.0
    if fill_a <= 0 or fill_b <= 0:
        return 0.0
    shares_a = 1.0 / fill_a
    shares_b = h / fill_b
    if trade.side == "short_spread":
        gross = shares_a * (fill_a - exit_a) + shares_b * (exit_b - fill_b)
    else:
        gross = shares_a * (exit_a - fill_a) + shares_b * (fill_b - exit_b)
    return gross - trade.friction_cost - trade.borrow_cost


def _trades_for_pair(
    ticker_a: str,
    ticker_b: str,
    prices: PricePanel,
    trade_slice: slice,
    intercept: float,
    hedge: float,
    config: BacktestConfig,
) -> list[Trade]:
    # Trading Period only. intercept/hedge already frozen from formation.
    close_a = np.log(prices.close[ticker_a].iloc[trade_slice].to_numpy(dtype=float))
    close_b = np.log(prices.close[ticker_b].iloc[trade_slice].to_numpy(dtype=float))
    trade_index = prices.close.index[trade_slice]
    # Spread from log(close). Open stays raw $ for the Fill (next session).
    spread = pd.Series(close_a - intercept - hedge * close_b, index=trade_index)
    rolling = spread.rolling(config.zscore_window)  # e.g. last 60 days, recompute daily
    roll_std = rolling.std(ddof=0)
    # Near-zero residual (exact hedge / float OLS) => |Z| explodes; treat as no signal.
    zscore = (spread - rolling.mean()) / roll_std
    zscore = zscore.where(roll_std > 1e-8)

    trades: list[Trade] = []
    pending_side: str | None = None  # signal today close -> Fill tomorrow open
    pending_exit_reason: str | None = None
    open_trade: Trade | None = None  # in a position until we flatten
    prev_z: float | None = None
    for date, z_raw in zscore.items():
        z = float(z_raw) if np.isfinite(z_raw) else float("nan")
        # 1) Execute yesterday's decision at TODAY's open (not yesterday's close)
        if pending_side is not None:
            fill_open_a = float(prices.open.loc[date, ticker_a])
            fill_open_b = float(prices.open.loc[date, ticker_b])
            if fill_open_a <= 0 or fill_open_b <= 0:
                pending_side = None
            else:
                open_trade = Trade(
                    ticker_a=ticker_a,
                    ticker_b=ticker_b,
                    fill_date=date,
                    fill_open_a=fill_open_a,
                    fill_open_b=fill_open_b,
                    side=pending_side,
                    hedge_ratio=hedge,
                    friction_cost=_turnover_friction(config, hedge),
                )
                trades.append(open_trade)
                pending_side = None
        elif pending_exit_reason is not None and open_trade is not None:
            open_trade.exit_date = date
            open_trade.exit_open_a = float(prices.open.loc[date, ticker_a])
            open_trade.exit_open_b = float(prices.open.loc[date, ticker_b])
            open_trade.exit_reason = pending_exit_reason
            open_trade.friction_cost += _turnover_friction(config, hedge)  # pay again at exit Fill
            open_trade.borrow_cost = _borrow_cost(
                config, open_trade.side, hedge, open_trade.fill_date, date
            )
            open_trade.pnl = _round_trip_pnl(open_trade)
            pending_exit_reason = None
            open_trade = None
        loc = trade_index.get_loc(date)
        has_next = loc + 1 < len(trade_index)  # need a next open to Fill
        # 2) After close: maybe queue an exit / entry for tomorrow
        if (
            open_trade is not None
            and pending_exit_reason is None
            and has_next
            and loc + 1 >= config.adf_window
        ):
            window = spread.iloc[loc + 1 - config.adf_window : loc + 1]
            # Constant / singular window: adfuller raises (same class as tiny-std Z).
            try:
                adf_p = adfuller(window.to_numpy(dtype=float), autolag="AIC")[1]
            except (ValueError, np.linalg.LinAlgError):
                adf_p = None
            if adf_p is not None and adf_p >= config.coint_pvalue:
                pending_exit_reason = "statistical_exit"
        if (
            open_trade is not None
            and pending_exit_reason is None
            and has_next
            and prev_z is not None
            and np.isfinite(z)
            and ((prev_z > 0 and z <= 0) or (prev_z < 0 and z >= 0))
        ):
            pending_exit_reason = "zscore_target"  # Spread came back through 0
        elif (
            open_trade is None
            and pending_side is None
            and has_next
            and np.isfinite(z)
            and abs(z) >= config.zscore_entry
        ):
            # |Z|>=2 at close. Open tomorrow can be anything; we still Fill.
            pending_side = "short_spread" if z >= config.zscore_entry else "long_spread"
        if np.isfinite(z):
            prev_z = z
    # Still in a trade when the 6-month window ends -> flatten last open
    if open_trade is not None and open_trade.exit_date is None:
        last = trade_index[-1]
        open_trade.exit_date = last
        open_trade.exit_open_a = float(prices.open.loc[last, ticker_a])
        open_trade.exit_open_b = float(prices.open.loc[last, ticker_b])
        open_trade.exit_reason = pending_exit_reason or "period_end"
        open_trade.friction_cost += _turnover_friction(config, hedge)  # period-end is an exit Fill too
        open_trade.borrow_cost = _borrow_cost(
            config, open_trade.side, hedge, open_trade.fill_date, last
        )
        open_trade.pnl = _round_trip_pnl(open_trade)
    return trades
