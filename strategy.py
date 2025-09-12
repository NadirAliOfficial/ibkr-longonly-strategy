import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from datetime import datetime
import re
import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv
from ib_insync import IB, Stock, util, BarDataList

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
    """
    Normalize any datetime-like Series/Index to ET tz-aware.
    Works if input is naive or already tz-aware.
    """
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
    Robust wrapper around reqHistoricalData:
      - retries with exponential backoff
      - prints progress when verbose
      - returns empty DataFrame on failure
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
        # cooperative sleep (ib_insync)
        delay = min(30.0, sleep_base * (2 ** attempt))
        ib.sleep(delay)
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
    out = np.empty(len(closes))
    out[:] = np.nan
    if len(closes) < period:
        return pd.Series(out, index=closes.index)

    # seed with SMA
    sma_seed = closes.iloc[:period].mean()
    out[period - 1] = sma_seed

    # recursive
    ema = sma_seed
    for i in range(period, len(closes)):
        price = closes.iloc[i]
        ema = (price * k) + (ema * (1 - k))
        out[i] = ema

    return pd.Series(out, index=closes.index)

def cmf_rolling_20(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series) -> pd.Series:
    """
    CMF(20) per bar: CMF = sum(MFV[20]) / sum(Vol[20]),
    where MFM = ((C-L) - (H-C)) / (H-L), MFV = MFM * V
    """
    h, l, c, v = h.astype(float), l.astype(float), c.astype(float), v.astype(float)
    rng = h - l
    with np.errstate(divide='ignore', invalid='ignore'):
        mfm = np.where(rng == 0, 0.0, ((c - l) - (h - c)) / rng)
    mfv = mfm * v.values
    mfv_s = pd.Series(mfv, index=h.index)
    cmf = mfv_s.rolling(20, min_periods=20).sum() / v.rolling(20, min_periods=20).sum()
    return cmf

# ----------------------------
# Data Fetch (IBKR) + Fallbacks
# ----------------------------
def fetch_daily_bars(ib: IB, contract: Stock, years: int, *, what: str, max_retries: int, verbose: bool) -> pd.DataFrame:
    """Fetch `years` of daily bars; if TRADES empty, fallback to MIDPOINT."""
    df = safe_hist(
        ib, contract,
        endDateTime="", durationStr=f"{years} Y",
        barSizeSetting="1 day", whatToShow=what, useRTH=True,
        max_retries=max_retries, verbose=verbose
    )
    if df.empty and what == "TRADES":
        if verbose:
            print("[IBKR] daily TRADES empty -> trying MIDPOINT fallback")
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
    """
    Fetch ~`years` of 5m RTH bars in chunks to avoid timeouts/pacing.
    Uses timezone-explicit endDateTime to avoid IB 2174 warnings.
    """
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
            if verbose:
                print("[IBKR] 5m TRADES empty -> MIDPOINT fallback this chunk")
            df_chunk = safe_hist(
                ib, contract,
                endDateTime=end, durationStr=chunk_str,
                barSizeSetting="5 mins", whatToShow="MIDPOINT", useRTH=True,
                max_retries=max_retries, verbose=verbose
            )

        if df_chunk.empty:
            if verbose:
                print("[IBKR] chunk empty; stopping pagination")
            break

        df_chunk["date"] = to_et(df_chunk["date"])
        df_chunk.set_index("date", inplace=True)
        df_chunk = df_chunk[["open", "high", "low", "close", "volume"]].copy()
        all_rows.append(df_chunk)

        # Step back to oldest bar for next request; use explicit UTC tz string to avoid 2174
        oldest = df_chunk.index.min()
        oldest_utc = oldest.tz_convert("UTC")
        end = oldest_utc.strftime("%Y%m%d-%H:%M:%S UTC")

        span_days = (df_chunk.index.max() - df_chunk.index.min()).days
        got_days += max(span_days, 1)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out

# ----------------------------
# Timeframe Utilities
# ----------------------------
def filter_rth_5m(df5: pd.DataFrame) -> pd.DataFrame:
    """Keep only 5m bars within RTH (09:30–16:00 ET)."""
    local_times = df5.index.tz_convert(ET)
    mask = (local_times.time >= RTH_START) & (local_times.time <= RTH_END)
    return df5.loc[mask].copy()

def _ohlcv_from_slice(end_ts: pd.Timestamp, sl: pd.DataFrame) -> dict:
    return {
        "end": float(end_ts.value),  # preserve ordering-friendly index later
        "open": float(sl["open"].iloc[0]),
        "high": float(sl["high"].max()),
        "low": float(sl["low"].min()),
        "close": float(sl["close"].iloc[-1]),
        "volume": float(sl["volume"].sum()),
        "et_index": end_ts,  # real ET timestamp
    }

def aggregate_rth_to_4h(df5_rth: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, pd.DataFrame]]:
    """
    Aggregate RTH (5m) to anchored blocks per session:
      - 09:30–13:30 ET (4h)
      - 13:30–16:00 ET (~2.5h)  [treated as a '4h block' for CMF rolling]
    Returns:
      df4h       : OHLCV by block end time
      block_map  : dict(block_end_ts -> 5m slice) for intrabar SL checks
    """
    if df5_rth.empty:
        return pd.DataFrame(), {}

    df5 = df5_rth.copy()
    df5["date_only"] = df5.index.tz_convert(ET).date
    df5["et_time"] = df5.index.tz_convert(ET).time

    blocks = []
    block_map: Dict[pd.Timestamp, pd.DataFrame] = {}

    for _, day_df in df5.groupby("date_only"):
        # First block: 09:30–13:30
        b1 = day_df[(day_df["et_time"] >= RTH_START) & (day_df["et_time"] <= RTH_MID)]
        if not b1.empty:
            end_ts = b1.index.max()
            block_map[end_ts] = b1.drop(columns=["date_only", "et_time"])
            blocks.append(_ohlcv_from_slice(end_ts, b1))

        # Second block: 13:30–16:00
        b2 = day_df[(day_df["et_time"] > RTH_MID) & (day_df["et_time"] <= RTH_END)]
        if not b2.empty:
            end_ts = b2.index.max()
            block_map[end_ts] = b2.drop(columns=["date_only", "et_time"])
            blocks.append(_ohlcv_from_slice(end_ts, b2))

    if not blocks:
        return pd.DataFrame(), {}

    df4h = pd.DataFrame(blocks).set_index("end")
    df4h.sort_index(inplace=True)
    return df4h, block_map

def restore_ts_index(df4h: pd.DataFrame) -> pd.DataFrame:
    """Convert the float 'end' index back to ET timestamps via stored 'et_index'."""
    df = df4h.copy()
    if "et_index" in df.columns:
        df.set_index("et_index", inplace=True)
        df.index.name = "end"
        df.drop(columns=["end"], inplace=True, errors="ignore")
    return df

def ema_value_before_bar(daily_ema: pd.Series, bar_end: pd.Timestamp) -> Optional[float]:
    """Use the most recent daily EMA strictly before that session's 16:00 ET close."""
    day = bar_end.tz_convert(ET).date()
    session_close = pd.Timestamp.combine(pd.Timestamp(day), RTH_END).tz_localize(ET)
    view = daily_ema[daily_ema.index < session_close]
    if view.empty:
        return None
    return float(view.iloc[-1])

# ----------------------------
# Trading Logic
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
    """
    Long-only backtest for one symbol. Returns (trades, equity_curve_df).
    Equity curve assumes 1 share per trade.
    """
    contract = Stock(symbol, "SMART", "USD")
    contract.primaryExchange = "NASDAQ"   # helps qualify many US stocks
    if verbose:
        print(f"[QUALIFY] {symbol} SMART/USD primary=NASDAQ")
    ib.qualifyContracts(contract)

    # Fetch data
    if verbose:
        print(f"[{symbol}] Fetching daily bars...")
    daily = fetch_daily_bars(ib, contract, years, what=what, max_retries=max_retries, verbose=verbose)

    if verbose:
        print(f"[{symbol}] Fetching 5m bars in chunks...")
    five = fetch_5m_bars_chunked(ib, contract, years, chunk_days=chunk_days, what=what, max_retries=max_retries, verbose=verbose)

    if daily.empty or five.empty:
        print(f"[{symbol}] No data returned from IBKR (check API login/permissions).")
        return [], pd.DataFrame()

    # RTH filter & aggregate to 4h blocks
    five_rth = filter_rth_5m(five)
    four_raw, block_map = aggregate_rth_to_4h(five_rth)
    if four_raw.empty:
        print(f"[{symbol}] No RTH 4h blocks.")
        return [], pd.DataFrame()
    four = restore_ts_index(four_raw)

    # Daily EMAs
    if verbose:
        print(f"[{symbol}] Computing daily EMA20/EMA30...")
    ema20_daily_full = ema_tv_seeded(daily["close"], 20)
    ema30_daily_full = ema_tv_seeded(daily["close"], 30)

    # CMF(20) on 4h blocks
    if verbose:
        print(f"[{symbol}] Computing CMF(20) on 4h blocks...")
    four["cmf20"] = cmf_rolling_20(four["high"], four["low"], four["close"], four["volume"])

    # Backtest state
    position_open = False
    entry_px: Optional[float] = None
    entry_idx: Optional[int] = None
    trades: List[Trade] = []
    cooldown_until = -1
    setup_window: List[bool] = []

    # Equity curve (1 share model)
    eq = 100_000.0
    equity_points = []
    idx_list = list(four.index)

    if verbose:
        print(f"[{symbol}] Running backtest over {len(four)} 4h blocks...")

    for i, (t, row) in enumerate(four.iterrows()):
        # cooldown gate
        if i < cooldown_until:
            equity_points.append({"time": t, "equity": eq})
            continue

        # resolve daily EMA20/30 to use (previous close)
        ema20_val = ema_value_before_bar(ema20_daily_full, t)
        ema30_val = ema_value_before_bar(ema30_daily_full, t)

        # Entry setup on current 4h block
        setup_ok = False
        cmf_ok = (not math.isnan(row["cmf20"])) and (row["cmf20"] > cmf_threshold)
        if (ema20_val is not None) and cmf_ok:
            setup_ok = (row["close"] > ema20_val)

        setup_window.append(bool(setup_ok))
        if len(setup_window) > cmf_confirm_bars:
            setup_window.pop(0)

        # ----- Exit logic (if in position) -----
        if position_open and entry_px is not None and entry_idx is not None:
            # Intrabar SL (armed after sl_arm_bars)
            if i >= entry_idx + sl_arm_bars:
                stop_px = entry_px * (1.0 - sl_pct)
                fast = block_map.get(t)
                if fast is not None and not fast.empty:
                    if float(fast["low"].min()) <= stop_px:
                        # SL intrabar
                        exit_px = float(stop_px)
                        eq *= (exit_px / entry_px)
                        trades.append(Trade(symbol, idx_list[entry_idx], entry_px, t, exit_px, "SL_FIXED_INTRABAR", i - entry_idx))
                        position_open, entry_px, entry_idx = False, None, None
                        cooldown_until = i + cooldown_bars + 1
                        equity_points.append({"time": t, "equity": eq})
                        continue

            # TP (profit-only) using chosen EMA
            current_px = float(row["close"])
            if current_px > entry_px:
                exit_line = None
                if exit_mode == "DAILY_EMA30":
                    exit_line = ema30_val
                elif exit_mode == "DAILY_EMA20":
                    exit_line = ema20_val
                if exit_line is not None and current_px < float(exit_line):
                    exit_px = current_px
                    eq *= (exit_px / entry_px)
                    trades.append(Trade(symbol, idx_list[entry_idx], entry_px, t, exit_px, f"TP_{exit_mode}", i - entry_idx))
                    position_open, entry_px, entry_idx = False, None, None
                    cooldown_until = i + cooldown_bars + 1
                    equity_points.append({"time": t, "equity": eq})
                    continue

            # End-of-bar SL
            stop_px = entry_px * (1.0 - sl_pct)
            if float(row["close"]) <= stop_px:
                exit_px = float(row["close"])
                eq *= (exit_px / entry_px)
                trades.append(Trade(symbol, idx_list[entry_idx], entry_px, t, exit_px, "SL_FIXED_4H", i - entry_idx))
                position_open, entry_px, entry_idx = False, None, None
                cooldown_until = i + cooldown_bars + 1
                equity_points.append({"time": t, "equity": eq})
                continue

        # ----- Entry logic (if flat) -----
        if not position_open:
            if len(setup_window) == cmf_confirm_bars and all(setup_window):
                # Enter at 4h close
                entry_px = float(row["close"])
                entry_idx = i
                position_open = True
                setup_window.clear()  # avoid immediate retrigger

        equity_points.append({"time": t, "equity": eq})

    # Optional: close open trade at last bar
    if position_open and entry_px is not None:
        last_t = four.index[-1]
        last_px = float(four["close"].iloc[-1])
        eq *= (last_px / entry_px)
        trades.append(Trade(symbol, idx_list[entry_idx], entry_px, last_t, last_px, "FORCE_EXIT_EOD", len(four) - entry_idx))

    equity_df = pd.DataFrame(equity_points).set_index("time")
    return trades, equity_df

# ----------------------------
# Outputs, Filenames & Stats
# ----------------------------
def build_filename(kind: str, symbol: str, args) -> str:
    # Compact, cross-platform friendly timestamp
    ts = datetime.now(ET).strftime("%Y%m%d_%H%M")

    # Descriptive parts without illegal characters
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

    # Strip characters that are invalid on Windows & co.
    safe = re.sub(r'[<>:"/\\|?*\n\r\t]+', "_", name).strip("_ ")

    return f"{safe}.csv"

def save_trades_csv(symbol: str, trades: List[Trade], args) -> None:
    path = build_filename("trades", symbol, args)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "entry_time", "entry_price", "exit_time", "exit_price", "reason", "bars_held", "return_pct"])
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
    if not trades:
        return "No trades."
    # Trade-based stats
    pnl = [(t.exit_price / t.entry_price - 1.0) for t in trades]
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x <= 0]
    win_rate = (len(wins) / len(pnl)) * 100.0 if pnl else 0.0
    pf = (sum(wins) / abs(sum(losses))) if sum(losses) != 0 else float("inf")
    avg_ret_pct = np.mean(pnl) * 100.0 if pnl else 0.0
    best = np.max(pnl) * 100.0
    worst = np.min(pnl) * 100.0

    # Equity-based stats (Cum P/L and Max Drawdown)
    if equity_df is not None and not equity_df.empty and "equity" in equity_df.columns:
        eq = equity_df["equity"].astype(float)
        eq0, eqN = float(eq.iloc[0]), float(eq.iloc[-1])
        cum_pl = eqN - eq0
        cum_pl_pct = (eqN / eq0 - 1.0) * 100.0
        run_max = eq.cummax()
        dd_series = eq / run_max - 1.0
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
    p = argparse.ArgumentParser(description="Backtest CMF(20) + Daily EMA20/30 strategy via IBKR")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID)
    p.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. AAPL,MSFT,SPY")
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--confirm-bars", type=int, default=2, help="Consecutive 4h bars required for entry")
    p.add_argument("--sl-pct", type=float, default=0.02)
    p.add_argument("--sl-arm-bars", type=int, default=2)
    p.add_argument("--exit-mode", choices=["DAILY_EMA20", "DAILY_EMA30"], default="DAILY_EMA30")
    p.add_argument("--cooldown-bars", type=int, default=1)
    p.add_argument("--chunk-days", type=int, default=90, help="Days per 5m chunk request (smaller = safer)")
    p.add_argument("--what", choices=["TRADES", "MIDPOINT"], default="TRADES", help="Primary whatToShow")
    p.add_argument("--cmf-threshold", type=float, default=0.0, help="CMF threshold; default 0.0 (was >0)")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    ib = IB()

    # surface IB errors immediately
    def on_error(reqId, errorCode, errorString, contract):
        print(f"[IBERR] code={errorCode} msg={errorString} reqId={reqId}")
    ib.errorEvent += on_error

    print(f"[CONNECT] {args.host}:{args.port} clientId={args.client_id}")
    ib.connect(args.host, args.port, clientId=args.client_id)

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

if __name__ == "__main__":
    main()
