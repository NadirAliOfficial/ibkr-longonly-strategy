import argparse
import asyncio
import time
import csv
import math
import os
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv
from ib_insync import IB, Stock, Crypto, util, BarDataList, MarketOrder, LimitOrder

# ----------------------------
# Config & CLI
# ----------------------------
load_dotenv()
DEFAULT_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("IBKR_PORT", "7497"))
DEFAULT_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "1"))

ET = pytz.timezone("America/New_York")
ET_TZ = "America/New_York"
RTH_START = pd.Timestamp("09:30", tz=ET).time()
RTH_MID   = pd.Timestamp("13:30", tz=ET).time()
RTH_END   = pd.Timestamp("16:00", tz=ET).time()

# ----------------------------
# TZ + Robust IBKR Helpers
# ----------------------------
def to_et(x):
    """Normalize any datetime-like Series/Index to ET tz-aware."""
    obj = pd.to_datetime(x, utc=True)
    if isinstance(obj, pd.DatetimeIndex):
        return obj.tz_convert(ET_TZ)
    elif isinstance(obj, pd.Series):
        return obj.dt.tz_convert(ET_TZ)
    else:
        return pd.DatetimeIndex(obj).tz_convert(ET_TZ)

def safe_hist(
    ib: IB,
    contract: Stock,
    *,
    endDateTime,  # str or None
    durationStr: str,
    barSizeSetting: str,
    whatToShow: str = "TRADES",
    useRTH: bool = True,
    max_retries: int = 5,
    sleep_base: float = 1.5,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Robust wrapper around reqHistoricalData (SYNC) — used by backtest only.
    """
    for attempt in range(max_retries):
        try:
            if verbose:
                end_str = ("NOW" if (endDateTime in ("", None)) else str(endDateTime))
                print(f"[IBKR] hist req attempt {attempt+1}/{max_retries} | "
                      f"{barSizeSetting} {durationStr} {whatToShow} RTH={useRTH} end='{end_str}'")
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=endDateTime,
                durationStr=durationStr,
                barSizeSetting=barSizeSetting,
                whatToShow=whatToShow,
                useRTH=useRTH,
                formatDate=1
            )
            if bars and len(bars) > 0:
                df = util.df(bars)
                if df is not None and not df.empty:
                    if verbose:
                        _min = pd.to_datetime(df["date"].min())
                        _max = pd.to_datetime(df["date"].max())
                        print(f"[IBKR] got {len(df)} rows [{_min} .. {_max}]")
                    return df
                else:
                    if verbose:
                        print("[IBKR] empty df returned")
        except Exception as e:
            if verbose:
                print(f"[IBKR] req exception: {e}")
        time.sleep(min(30.0, sleep_base * (2 ** attempt)))
    return pd.DataFrame()

# ----------------------------
# Indicators
# ----------------------------
def ema_tv_seeded(closes: pd.Series, period: int) -> pd.Series:
    """
    TradingView-consistent EMA:
      - seed with SMA of first N closes
      - recursive k = 2/(N+1)
    """
    closes = closes.dropna().astype(float)
    if len(closes) == 0:
        return pd.Series(dtype=float, index=closes.index)

    k = 2 / (period + 1)
    out = np.empty(len(closes)); out[:] = np.nan
    if len(closes) < period:
        return pd.Series(out, index=closes.index)

    sma_seed = closes.iloc[:period].mean()
    out[period - 1] = sma_seed

    ema = sma_seed
    for i in range(period, len(closes)):
        price = closes.iloc[i]
        ema = (price * k) + (ema * (1 - k))
        out[i] = ema

    return pd.Series(out, index=closes.index)

def cmf_rolling_20(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series) -> pd.Series:
    """CMF(20) = sum(MFV[20]) / sum(Vol[20]) with MFM=((C-L)-(H-C))/(H-L), MFV=MFM*V."""
    h, l, c, v = h.astype(float), l.astype(float), c.astype(float), v.astype(float)
    rng = h - l
    with np.errstate(divide='ignore', invalid='ignore'):
        mfm = np.where(rng == 0, 0.0, ((c - l) - (h - c)) / rng)
    mfv = mfm * v.values
    mfv_s = pd.Series(mfv, index=h.index)
    return mfv_s.rolling(20, min_periods=20).sum() / v.rolling(20, min_periods=20).sum()

# ----------------------------
# Data Fetch (Backtest) + Fallbacks
# ----------------------------
def fetch_daily_bars(ib: IB, contract: Stock, years: int, *, what: str, max_retries: int, verbose: bool) -> pd.DataFrame:
    """SYNC: Used only by backtest path."""
    df = safe_hist(
        ib, contract,
        endDateTime="", durationStr=f"{years} Y",
        barSizeSetting="1 day", whatToShow=what, useRTH=True,
        max_retries=max_retries, verbose=verbose
    )
    if df.empty and what == "TRADES":
        if verbose: print("[IBKR] daily TRADES empty -> trying MIDPOINT fallback")
        df = safe_hist(
            ib, contract,
            endDateTime="", durationStr=f"{years} Y",
            barSizeSetting="1 day", whatToShow="MIDPOINT", useRTH=True,
            max_retries=max_retries, verbose=verbose
        )
    if df.empty:
        return df
    df["date"] = to_et(df["date"])
    df.set_index("date", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]

def fetch_5m_bars_chunked(
    ib: IB,
    contract: Stock,
    years: int,
    *,
    chunk_days: int,
    what: str,
    max_retries: int,
    verbose: bool
) -> pd.DataFrame:
    """SYNC: Backtest 5m fetch in chunks; avoids pacing; explicit UTC end to dodge 2174."""
    all_rows: List[pd.DataFrame] = []
    end = ""  # now
    target_days = years * 365
    got_days = 0
    chunk_str = f"{chunk_days} D"

    while got_days < target_days:
        if verbose:
            print(f"[IBKR] fetch 5m chunk (target_days={target_days}, got_days={got_days}) end={end or 'NOW'}")
        df_chunk = safe_hist(
            ib, contract,
            endDateTime=end, durationStr=chunk_str,
            barSizeSetting="5 mins", whatToShow=what, useRTH=True,
            max_retries=max_retries, verbose=verbose
        )
        if df_chunk.empty and what == "TRADES":
            if verbose: print("[IBKR] 5m TRADES empty -> MIDPOINT fallback this chunk")
            df_chunk = safe_hist(
                ib, contract,
                endDateTime=end, durationStr=chunk_str,
                barSizeSetting="5 mins", whatToShow="MIDPOINT", useRTH=True,
                max_retries=max_retries, verbose=verbose
            )

        if df_chunk.empty:
            if verbose: print("[IBKR] chunk empty; stopping pagination")
            break

        df_chunk["date"] = to_et(df_chunk["date"])
        df_chunk.set_index("date", inplace=True)
        df_chunk = df_chunk[["open", "high", "low", "close", "volume"]].copy()
        all_rows.append(df_chunk)

        oldest = df_chunk.index.min().tz_convert("UTC")
        end = oldest.strftime("%Y%m%d-%H:%M:%S")   # no UTC suffix


        span_days = (df_chunk.index.max() - df_chunk.index.min()).days
        got_days += max(span_days, 1)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows).sort_index()
    return out[~out.index.duplicated(keep="first")]

# ----------------------------
# Timeframe Utilities
# ----------------------------
def is_rth_now(et_dt=None) -> bool:
    et_dt = et_dt or datetime.now(ET)
    t = et_dt.time()
    return et_dt.weekday() < 5 and (RTH_START <= t <= RTH_END)

def at_block_close(et_ts: pd.Timestamp) -> bool:
    tt = et_ts.tz_convert(ET).time()
    return tt == RTH_MID or tt == RTH_END

def filter_rth_5m(df5: pd.DataFrame) -> pd.DataFrame:
    local_times = df5.index.tz_convert(ET)
    mask = (local_times.time >= RTH_START) & (local_times.time <= RTH_END)
    return df5.loc[mask].copy()

def _ohlcv_from_slice(end_ts: pd.Timestamp, sl: pd.DataFrame) -> dict:
    return {
        "end": float(end_ts.value),
        "open": float(sl["open"].iloc[0]),
        "high": float(sl["high"].max()),
        "low": float(sl["low"].min()),
        "close": float(sl["close"].iloc[-1]),
        "volume": float(sl["volume"].sum()),
        "et_index": end_ts,
    }

def aggregate_rth_to_4h(df5_rth: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, pd.DataFrame]]:
    """Aggregate to [09:30–13:30] and [13:30–16:00] ET blocks."""
    if df5_rth.empty:
        return pd.DataFrame(), {}
    df5 = df5_rth.copy()
    df5["date_only"] = df5.index.tz_convert(ET).date
    df5["et_time"] = df5.index.tz_convert(ET).time

    blocks, block_map = [], {}
    for _, day_df in df5.groupby("date_only"):
        b1 = day_df[(day_df["et_time"] >= RTH_START) & (day_df["et_time"] <= RTH_MID)]
        if not b1.empty:
            end_ts = b1.index.max()
            block_map[end_ts] = b1.drop(columns=["date_only", "et_time"])
            blocks.append(_ohlcv_from_slice(end_ts, b1))
        b2 = day_df[(day_df["et_time"] > RTH_MID) & (day_df["et_time"] <= RTH_END)]
        if not b2.empty:
            end_ts = b2.index.max()
            block_map[end_ts] = b2.drop(columns=["date_only", "et_time"])
            blocks.append(_ohlcv_from_slice(end_ts, b2))

    if not blocks:
        return pd.DataFrame(), {}
    df4h = pd.DataFrame(blocks).set_index("end").sort_index()
    return df4h, block_map

def restore_ts_index(df4h: pd.DataFrame) -> pd.DataFrame:
    df = df4h.copy()
    if "et_index" in df.columns:
        df.set_index("et_index", inplace=True)
        df.index.name = "end"
        df.drop(columns=["end"], inplace=True, errors="ignore")
    return df

def ema_value_before_bar(daily_ema: pd.Series, bar_end: pd.Timestamp) -> Optional[float]:
    day = bar_end.tz_convert(ET).date()
    session_close = pd.Timestamp.combine(pd.Timestamp(day), RTH_END).tz_localize(ET)
    view = daily_ema[daily_ema.index < session_close]
    if view.empty:
        return None
    return float(view.iloc[-1])

# ----------------------------
# Trading Logic (Backtest)
# ----------------------------
@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    reason: str
    bars_held: int

def run_backtest_for_symbol(
    ib: IB,
    symbol: str,
    years: int = 10,
    cmf_confirm_bars: int = 2,
    sl_pct: float = 0.02,
    sl_arm_bars: int = 2,
    exit_mode: str = "DAILY_EMA30",
    cooldown_bars: int = 1,
    *,
    what: str,
    chunk_days: int,
    max_retries: int,
    verbose: bool,
    cmf_threshold: float = 0.0
) -> Tuple[List[Trade], pd.DataFrame]:
    contract = Stock(symbol, "SMART", "USD"); contract.primaryExchange = "NASDAQ"
    if verbose: print(f"[QUALIFY] {symbol} SMART/USD primary=NASDAQ")
    ib.qualifyContracts(contract)

    if verbose: print(f"[{symbol}] Fetching daily bars...")
    daily = fetch_daily_bars(ib, contract, years, what=what, max_retries=max_retries, verbose=verbose)

    if verbose: print(f"[{symbol}] Fetching 5m bars in chunks...")
    five = fetch_5m_bars_chunked(ib, contract, years, chunk_days=chunk_days, what=what, max_retries=max_retries, verbose=verbose)

    if daily.empty or five.empty:
        print(f"[{symbol}] No data returned from IBKR (check API login/permissions).")
        return [], pd.DataFrame()

    five_rth = filter_rth_5m(five)
    four_raw, block_map = aggregate_rth_to_4h(five_rth)
    if four_raw.empty:
        print(f"[{symbol}] No RTH 4h blocks."); return [], pd.DataFrame()
    four = restore_ts_index(four_raw)

    if verbose: print(f"[{symbol}] Computing daily EMA20/EMA30...")
    ema20_daily_full = ema_tv_seeded(daily["close"], 20)
    ema30_daily_full = ema_tv_seeded(daily["close"], 30)

    if verbose: print(f"[{symbol}] Computing CMF(20) on 4h blocks...")
    four["cmf20"] = cmf_rolling_20(four["high"], four["low"], four["close"], four["volume"])

    position_open = False
    entry_px: Optional[float] = None
    entry_idx: Optional[int] = None
    trades: List[Trade] = []
    cooldown_until = -1
    setup_window: List[bool] = []

    eq = 100_000.0
    equity_points = []
    idx_list = list(four.index)

    if verbose: print(f"[{symbol}] Running backtest over {len(four)} 4h blocks...")

    for i, (t, row) in enumerate(four.iterrows()):
        if i < cooldown_until:
            equity_points.append({"time": t, "equity": eq}); continue

        ema20_val = ema_value_before_bar(ema20_daily_full, t)
        ema30_val = ema_value_before_bar(ema30_daily_full, t)

        setup_ok = False
        cmf_ok = (not math.isnan(row["cmf20"])) and (row["cmf20"] > cmf_threshold)
        if (ema20_val is not None) and cmf_ok:
            setup_ok = (row["close"] > ema20_val)

        setup_window.append(bool(setup_ok))
        if len(setup_window) > cmf_confirm_bars:
            setup_window.pop(0)

        if position_open and entry_px is not None and entry_idx is not None:
            if i >= entry_idx + sl_arm_bars:
                stop_px = entry_px * (1.0 - sl_pct)
                fast = block_map.get(t)
                if fast is not None and not fast.empty:
                    if float(fast["low"].min()) <= stop_px:
                        exit_px = float(stop_px)
                        eq *= (exit_px / entry_px)
                        trades.append(Trade(symbol, idx_list[entry_idx], entry_px, t, exit_px, "SL_FIXED_INTRABAR", i - entry_idx))
                        position_open, entry_px, entry_idx = False, None, None
                        cooldown_until = i + cooldown_bars + 1
                        equity_points.append({"time": t, "equity": eq}); continue

            current_px = float(row["close"])
            if current_px > entry_px:
                exit_line = ema30_val if exit_mode == "DAILY_EMA30" else ema20_val
                if exit_line is not None and current_px < float(exit_line):
                    exit_px = current_px
                    eq *= (exit_px / entry_px)
                    trades.append(Trade(symbol, idx_list[entry_idx], entry_px, t, exit_px, f"TP_{exit_mode}", i - entry_idx))
                    position_open, entry_px, entry_idx = False, None, None
                    cooldown_until = i + cooldown_bars + 1
                    equity_points.append({"time": t, "equity": eq}); continue

            stop_px = entry_px * (1.0 - sl_pct)
            if float(row["close"]) <= stop_px:
                exit_px = float(row["close"])
                eq *= (exit_px / entry_px)
                trades.append(Trade(symbol, idx_list[entry_idx], entry_px, t, exit_px, "SL_FIXED_4H", i - entry_idx))
                position_open, entry_px, entry_idx = False, None, None
                cooldown_until = i + cooldown_bars + 1
                equity_points.append({"time": t, "equity": eq}); continue

        if not position_open:
            if cmf_confirm_bars == 0 or (len(setup_window) == cmf_confirm_bars and all(setup_window)):
                entry_px = float(row["close"]); entry_idx = i; position_open = True; setup_window.clear()

        equity_points.append({"time": t, "equity": eq})

    if position_open and entry_px is not None:
        last_t = four.index[-1]; last_px = float(four["close"].iloc[-1])
        eq *= (last_px / entry_px)
        trades.append(Trade(symbol, idx_list[entry_idx], entry_px, last_t, last_px, "FORCE_EXIT_EOD", len(four) - entry_idx))

    equity_df = pd.DataFrame(equity_points).set_index("time")
    plot_indicators(four, ema20_daily_full, ema30_daily_full)

    return trades, equity_df

# ----------------------------
# LIVE Helpers (ASYNC)
# ----------------------------
async def fetch_daily_bars_async(
    ib: IB, contract: Stock, years: int, *,
    what: str, max_retries: int, verbose: bool
) -> pd.DataFrame:
    """Async daily history fetch for LIVE (avoids nested loops)."""
    for attempt in range(max_retries):
        try:
            if verbose:
                print(f"[IBKR-ASYNC] daily attempt {attempt+1}/{max_retries} | 1 day {years} Y {what}")
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=f"{years} Y",
                barSizeSetting="1 day",
                whatToShow=what,
                useRTH=True,
                formatDate=1
            )
            if bars:
                df = util.df(bars)
                if df is not None and not df.empty:
                    df["date"] = to_et(df["date"])
                    df.set_index("date", inplace=True)
                    return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            if verbose:
                print(f"[IBKR-ASYNC] daily exception: {e}")
        await asyncio.sleep(min(30.0, 1.5 * (2 ** attempt)))


    if what == "TRADES":
        # fallback to MIDPOINT
        try:
            if verbose: print("[IBKR-ASYNC] fallback daily MIDPOINT")
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=f"{years} Y",
                barSizeSetting="1 day",
                whatToShow="MIDPOINT",
                useRTH=True,
                formatDate=1
            )
            if bars:
                df = util.df(bars)
                if df is not None and not df.empty:
                    df["date"] = to_et(df["date"])
                    df.set_index("date", inplace=True)
                    return df[["open", "high", "low", "close", "volume"]]
        except Exception:
            pass
    return pd.DataFrame()

# ----------------------------
# LIVE Runner (ASYNC-SAFE)
# ----------------------------
class LiveSymbol:
    """
    Minimal live runner per symbol:
      - Streams 5m RTH bars (keepUpToDate)
      - Aggregates to 2 blocks/day (09:30–13:30, 13:30–16:00)
      - Entry: price > daily EMA20 & CMF(20) > threshold with N confirmations
      - TP: profit-only & price < chosen daily EMA (20/30)
      - SL: armed after sl_arm_bars blocks; intrabar triggers on 5m low <= stop
      - No pyramiding; simple cooldown in bars
    """
    def __init__(self, ib: IB, symbol: str, args):
        self.ib = ib
        self.args = args
        self.symbol = symbol
        if symbol.upper() in ("BTCUSD", "ETHUSD"):
            # Create Crypto contract via PAXOS
            base = symbol.replace("USD", "")
            self.contract = Crypto(base, "PAXOS")
        else:
            # Default to Stock contract
            self.contract = Stock(symbol, "SMART", "USD")
            self.contract.primaryExchange = "NASDAQ"

        self.position_open = False
        self.entry_px: Optional[float] = None
        self.entry_block_index: Optional[int] = None
        self.cooldown_until = -1
        self.confirm_q = deque(maxlen=args.confirm_bars)
        self.blocks: List[dict] = []
        self.block_df = pd.DataFrame()
        self.block_map: Dict[pd.Timestamp, pd.DataFrame] = {}

        self._current_day = None
        self._morning: List[pd.DataFrame] = []
        self._afternoon: List[pd.DataFrame] = []

        self.ema20_daily: Optional[pd.Series] = None
        self.ema30_daily: Optional[pd.Series] = None

    async def _resolve_px(self, last5m_close: float) -> float:
        # snapshot mkt data; give it a brief moment to populate
        t = self.ib.reqMktData(self.contract, snapshot=True, regulatorySnapshot=False)
        await asyncio.sleep(0.25)
        px = None
        if t is not None:
            try:
                px = t.midpoint() or t.last or t.close
            except Exception:
                px = None
        return float(px) if px else float(last5m_close)

    def _finalize_block(self, slice_df: pd.DataFrame) -> dict:
        end_ts = slice_df.index.max()
        return {
            "end": float(end_ts.value),
            "open": float(slice_df["open"].iloc[0]),
            "high": float(slice_df["high"].max()),
            "low": float(slice_df["low"].min()),
            "close": float(slice_df["close"].iloc[-1]),
            "volume": float(slice_df["volume"].sum()),
            "et_index": end_ts,
        }

    def _cmf20_latest(self) -> Optional[float]:
        if self.block_df.empty or len(self.block_df) < 20:
            return None
        return float(cmf_rolling_20(
            self.block_df["high"], self.block_df["low"],
            self.block_df["close"], self.block_df["volume"]
        ).iloc[-1])

    def _ema_before(self, ema_series: pd.Series, bar_end: pd.Timestamp) -> Optional[float]:
        return ema_value_before_bar(ema_series, bar_end)

    async def _recalc_daily_emas(self):
        daily = await fetch_daily_bars_async(
            self.ib, self.contract, self.args.years,
            what=self.args.what_daily, max_retries=self.args.max_retries, verbose=self.args.verbose
        )
        if daily.empty:
            print(f"[{self.symbol}] WARNING: daily bars empty; EMA refresh skipped"); return
        self.ema20_daily = ema_tv_seeded(daily["close"], 20)
        self.ema30_daily = ema_tv_seeded(daily["close"], 30)
        if self.args.verbose:
            print(f"[{self.symbol}] Daily EMA20/30 refreshed")

    def _place_market(self, side: str, qty: int, note: str):
        order = MarketOrder(side, qty)
        if self.args.account: order.account = self.args.account
        self.ib.placeOrder(self.contract, order)
        print(f"[{self.symbol}] {note}")

    def _place_limit(self, side: str, qty: int, price: float, note: str):
        order = LimitOrder(side, qty, lmtPrice=price, tif="DAY")
        if self.args.account: order.account = self.args.account
        self.ib.placeOrder(self.contract, order)
        print(f"[{self.symbol}] {note}")

    def _maybe_exit(self, now_ts: pd.Timestamp, last5m_low: float, last5m_close: float, current_block_i: int):
        if not self.position_open or self.entry_px is None or self.entry_block_index is None:
            return

        # SL armed after N blocks
        if current_block_i >= (self.entry_block_index + self.args.sl_arm_bars):
            stop_px = self.entry_px * (1.0 - self.args.sl_pct)
            if last5m_low <= stop_px:
                self._place_market("SELL", self.args.live_qty, f"EXIT SL_INTRABAR ~{stop_px:.4f}")
                self.position_open = False
                self.entry_px = None
                self.entry_block_index = None
                self.cooldown_until = current_block_i + self.args.cooldown_bars + 1
                return

        # TP (profit-only) vs EMA20/30
        if last5m_close > self.entry_px and self.ema20_daily is not None and self.ema30_daily is not None:
            line = self._ema_before(self.ema20_daily if self.args.exit_mode == "DAILY_EMA20" else self.ema30_daily, now_ts)
            if line is not None and last5m_close < float(line):
                self._place_market("SELL", self.args.live_qty, f"EXIT TP_{self.args.exit_mode} ~{last5m_close:.4f}")
                self.position_open = False
                self.entry_px = None
                self.entry_block_index = None
                self.cooldown_until = current_block_i + self.args.cooldown_bars + 1
                return

    async def _maybe_enter(self, now_ts: pd.Timestamp, last_close: float, current_block_i: int):
        if self.position_open or current_block_i < self.cooldown_until or self.ema20_daily is None:
            return
        ema20 = self._ema_before(self.ema20_daily, now_ts)
        if ema20 is None: return
        cmf = self._cmf20_latest()
        cmf_ok = (cmf is not None) and (cmf > self.args.cmf_threshold)
        setup_ok = cmf_ok and (last_close > float(ema20))
        self.confirm_q.append(bool(setup_ok))
        if self.args.confirm_bars > 0 and (len(self.confirm_q) < self.args.confirm_bars or not all(self.confirm_q)):
            return

        px = await self._resolve_px(last_close)
        self._place_limit("BUY", self.args.live_qty, px, f"ENTER BUY LMT {self.args.live_qty} @ {px:.4f}")
        self.position_open = True
        self.entry_px = px
        self.entry_block_index = current_block_i

    async def run(self):
        # Async contract qualify
        await self.ib.qualifyContractsAsync(self.contract)
        # Initial EMAs
        await self._recalc_daily_emas()

        # Stream 5m bars (keepUpToDate)
        bars: BarDataList = await self.ib.reqHistoricalDataAsync(
            self.contract,
            endDateTime="",
            durationStr="2 D",
            barSizeSetting="5 mins",
            whatToShow=self.args.what_5m,
            useRTH=False,
            formatDate=1,
            keepUpToDate=True
        )

        def bars_to_df(bdl: BarDataList) -> pd.DataFrame:
            if not bdl: return pd.DataFrame()
            df = util.df(bdl)
            if df is None or df.empty: return pd.DataFrame()
            df["date"] = to_et(df["date"])
            df.set_index("date", inplace=True)
            return df[["open","high","low","close","volume"]]

        last_len = 0
        while True:
            # Refresh EMAs a few mins after close
            now_et = datetime.now(ET)
            if now_et.hour == 16 and 5 <= now_et.minute < 10:
                await self._recalc_daily_emas()

            # Let ib_insync process socket events
            await asyncio.sleep(1.0)

            # If the stream grew, process the latest bar
            if len(bars) != last_len:
                last_len = len(bars)
                df5 = bars_to_df(bars)
                df5 = filter_rth_5m(df5)
                await self._on_new_5m_bar(df5)

    async def _on_new_5m_bar(self, df5_new: pd.DataFrame):
        if df5_new.empty: return
        bar_ts = df5_new.index.max()
        bar_ts_et = bar_ts.tz_convert(ET)
        if isinstance(self.contract, Stock):
            if not is_rth_now(bar_ts_et):
                return


        t = bar_ts_et.time()
        if self._current_day is None:
            self._current_day = bar_ts_et.date()
        if bar_ts_et.date() != self._current_day:
            self._current_day = bar_ts_et.date()
            self._morning, self._afternoon = [], []

        row = df5_new.loc[bar_ts]
        row_df = pd.DataFrame([row], index=[bar_ts])

        if RTH_START <= t <= RTH_MID:
            self._morning.append(row_df)
        elif RTH_MID < t <= RTH_END:
            self._afternoon.append(row_df)

        current_block_i = len(self.blocks) - 1 if self.blocks else -1
        self._maybe_exit(bar_ts, float(row["low"]), float(row["close"]), current_block_i)

        if at_block_close(bar_ts):
            if t == RTH_MID and self._morning:
                s = pd.concat(self._morning); rec = self._finalize_block(s)
                self.blocks.append(rec); self.block_map[rec["et_index"]] = s
                self.block_df = pd.DataFrame(self.blocks).set_index("et_index")
                print(f"[{self.symbol}] BLOCK 09:30–13:30 closed, blocks={len(self.block_df)}")
                await self._maybe_enter(bar_ts, float(rec["close"]), len(self.block_df)-1)

            if t == RTH_END and self._afternoon:
                s = pd.concat(self._afternoon); rec = self._finalize_block(s)
                self.blocks.append(rec); self.block_map[rec["et_index"]] = s
                self.block_df = pd.DataFrame(self.blocks).set_index("et_index")
                print(f"[{self.symbol}] BLOCK 13:30–16:00 closed, blocks={len(self.block_df)}")
                if self.position_open and self.entry_px is not None:
                    stop_px = self.entry_px * (1.0 - self.args.sl_pct)
                    if float(rec["close"]) <= stop_px:
                        self._place_market("SELL", self.args.live_qty, f"EXIT SL_4H ~{float(rec['close']):.4f}")
                        self.position_open = False
                        self.entry_px = None
                        self.entry_block_index = None
                        self.cooldown_until = len(self.block_df) - 1 + self.args.cooldown_bars + 1
                self._maybe_exit(bar_ts, float(s["low"].min()), float(rec["close"]), len(self.block_df)-1)

# ----------------------------
# Outputs, Filenames & Stats
# ----------------------------
def build_filename(kind: str, symbol: str, args) -> str:
    ts = datetime.now(ET).strftime("%Y%m%d_%H%M")
    parts = [
        f"{kind}_{symbol}",
        f"years-{args.years}",
        f"exit-{args.exit_mode}",
        f"slpct-{args.sl_pct}",
        f"confirm-{args.confirm_bars}",
        f"slarm-{args.sl_arm_bars}",
        f"cooldown-{args.cooldown_bars}",
        ts,
    ]
    name = " ".join(parts)
    safe = re.sub(r'[<>:\"/\\\\|?*\\n\\r\\t]+', "_", name).strip("_ ")
    return f"{safe}.csv"

def save_trades_csv(symbol: str, trades: List[Trade], args) -> None:
    path = build_filename("trades", symbol, args)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol","entry_time","entry_price","exit_time","exit_price","reason","bars_held","return_pct"])
        for tr in trades:
            ret_pct = (tr.exit_price / tr.entry_price - 1.0) * 100.0
            w.writerow([tr.symbol,
                        pd.Timestamp(tr.entry_time).isoformat(),
                        f"{tr.entry_price:.6f}",
                        pd.Timestamp(tr.exit_time).isoformat(),
                        f"{tr.exit_price:.6f}",
                        tr.reason,
                        tr.bars_held,
                        f"{ret_pct:.4f}"])
    print(f"[{symbol}] Saved {path}")

def save_equity_csv(symbol: str, equity_df: pd.DataFrame, args) -> None:
    path = build_filename("equity", symbol, args)
    equity_df.to_csv(path, float_format="%.6f")
    print(f"[{symbol}] Saved {path}")

def quick_stats(trades: List[Trade], equity_df: pd.DataFrame) -> str:
    if not trades: return "No trades."
    pnl = [(t.exit_price / t.entry_price - 1.0) for t in trades]
    wins = [x for x in pnl if x > 0]; losses = [x for x in pnl if x <= 0]
    win_rate = (len(wins) / len(pnl)) * 100.0 if pnl else 0.0
    pf = (sum(wins) / abs(sum(losses))) if sum(losses) != 0 else float("inf")
    avg_ret_pct = np.mean(pnl) * 100.0 if pnl else 0.0
    best = np.max(pnl) * 100.0; worst = np.min(pnl) * 100.0

    if equity_df is not None and not equity_df.empty and "equity" in equity_df.columns:
        eq = equity_df["equity"].astype(float)
        eq0, eqN = float(eq.iloc[0]), float(eq.iloc[-1])
        cum_pl = eqN - eq0; cum_pl_pct = (eqN / eq0 - 1.0) * 100.0
        run_max = eq.cummax(); dd_series = eq / run_max - 1.0
        max_dd_pct = float(dd_series.min() * 100.0)
    else:
        cum_pl, cum_pl_pct, max_dd_pct = 0.0, 0.0, 0.0

    return (f"Trades: {len(pnl)} | Win%: {win_rate:.1f} | AvgRet: {avg_ret_pct:.2f}% | "
            f"PF: {pf:.2f} | Best: {best:.2f}% | Worst: {worst:.2f}% | "
            f"Cum P/L: {cum_pl:+.2f} ({cum_pl_pct:+.2f}%) | MaxDD: {max_dd_pct:.2f}%")

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Backtest/Live CMF(20) + Daily EMA20/30 via IBKR")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    p.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. AAPL,MSFT,SPY")
    p.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    p.add_argument("--account", default=os.getenv("IBKR_ACCOUNT", None), help="IBKR account id, e.g. DU1234567")
    p.add_argument("--live-qty", type=int, default=int(os.getenv("LIVE_QTY", "1")), help="Order size per symbol (no pyramiding)")

    # Shared strategy params
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--confirm-bars", type=int, default=2, help="Consecutive 4h blocks required for entry")
    p.add_argument("--sl-pct", type=float, default=0.02)
    p.add_argument("--sl-arm-bars", type=int, default=2)
    p.add_argument("--exit-mode", choices=["DAILY_EMA20", "DAILY_EMA30"], default="DAILY_EMA30")
    p.add_argument("--cooldown-bars", type=int, default=1)
    p.add_argument("--cmf-threshold", type=float, default=0.0, help="CMF threshold; default 0.0 (was >0)")

    # Backtest data controls
    p.add_argument("--chunk-days", type=int, default=90, help="Days per 5m chunk request (smaller = safer)")
    p.add_argument("--what", choices=["TRADES", "MIDPOINT"], default="TRADES", help="Backtest whatToShow")

    # Live data controls
    p.add_argument("--what-daily", choices=["TRADES","MIDPOINT"], default="TRADES", help="whatToShow for daily EMAs in LIVE")
    p.add_argument("--what-5m", choices=["TRADES","MIDPOINT"], default="MIDPOINT", help="whatToShow for 5m stream in LIVE")

    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

#  Plotting 
def plot_indicators(four: pd.DataFrame, ema20_daily: pd.Series, ema30_daily: pd.Series):
    """
    Plot 4h OHLC + CMF(20) + daily EMA20/30 overlays.
    """
    # Map daily EMA values to 4h timestamps
    ema20_mapped = four.index.map(lambda ts: ema_value_before_bar(ema20_daily, ts))
    ema30_mapped = four.index.map(lambda ts: ema_value_before_bar(ema30_daily, ts))

    fig, (ax_price, ax_cmf) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                           gridspec_kw={'height_ratios': [3, 1]})

    # --- Price + EMAs ---
    ax_price.plot(four.index, four["close"], label="Close (4h)", color="black", linewidth=1)
    ax_price.plot(four.index, ema20_mapped, label="EMA20 (Daily)", color="blue", linewidth=1.2)
    ax_price.plot(four.index, ema30_mapped, label="EMA30 (Daily)", color="red", linewidth=1.2)
    ax_price.set_title("4h Close with EMA20/30")
    ax_price.legend(loc="upper left")
    ax_price.grid(True, alpha=0.3)

    # --- CMF(20) ---
    ax_cmf.plot(four.index, four["cmf20"], label="CMF(20)", color="green", linewidth=1)
    ax_cmf.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax_cmf.set_title("Chaikin Money Flow (CMF 20)")
    ax_cmf.legend(loc="upper left")
    ax_cmf.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    ib = IB()

    def on_error(reqId, errorCode, errorString, contract):
        print(f"[IBERR] code={errorCode} msg={errorString} reqId={reqId}")
    ib.errorEvent += on_error

    print(f"[CONNECT] {args.host}:{args.port} clientId={args.client_id}")
    ib.connect(args.host, args.port, clientId=args.client_id)

    if args.mode == "backtest":
        for sym in symbols:
            print(f"\n=== Backtesting {sym} ===")
            trades, equity = run_backtest_for_symbol(
                ib,
                sym,
                years=args.years,
                cmf_confirm_bars=args.confirm_bars,
                sl_pct=args.sl_pct,
                sl_arm_bars=args.sl_arm_bars,
                exit_mode=args.exit_mode,
                cooldown_bars=args.cooldown_bars,
                what=args.what,
                chunk_days=args.chunk_days,
                max_retries=args.max_retries,
                verbose=args.verbose,
                cmf_threshold=args.cmf_threshold
            )
            save_trades_csv(sym, trades, args)
            if not equity.empty:
                save_equity_csv(sym, equity, args)
            print(f"[{sym}] {quick_stats(trades, equity)}")
        ib.disconnect()
        return

    # LIVE MODE
    print("\n=== LIVE MODE ===")
    runners = [LiveSymbol(ib, s, args) for s in symbols]

    async def _run_all():
        await asyncio.gather(*(r.run() for r in runners))

    ib.run(_run_all())

if __name__ == "__main__":
    main()
