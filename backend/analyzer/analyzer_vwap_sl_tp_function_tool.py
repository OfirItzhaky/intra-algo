"""
VWAP SL/TP Function Toolkit
============================

This module provides six exit logic functions used across all 18 VWAP-based scalping strategies.
Each function computes the dynamic Stop Loss (SL) and Take Profit (TP) levels based on the strategy rules.

Input:
- df: pd.DataFrame containing the full OHLC and indicator context
- entry_index: int index of the entry bar
- entry_price: float price at which the trade was entered
- side: "long" or "short"
- additional kwargs based on rule type (e.g., atr_mult, r_multiple, etc.)

Output:
- dict with keys: "sl" (stop loss), "tp" (take profit)
"""

from typing import Dict, Optional
import pandas as pd
from logging_setup import get_logger

log = get_logger(__name__)

def calc_exit_by_r_multiple(
    df: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    side: str,
    r_multiple: float = 2.0,
    stop_ticks: Optional[float] = None,
) -> Dict[str, float]:
    """
    Exit based on fixed R-multiple. Stop is at a known distance, target is R× that.
    """
    if stop_ticks is None:
        raise ValueError("stop_ticks is required for R-multiple exits")

    if side == "long":
        sl = entry_price - stop_ticks
        tp = entry_price + r_multiple * stop_ticks
    else:
        sl = entry_price + stop_ticks
        tp = entry_price - r_multiple * stop_ticks

    return {"sl": sl, "tp": tp}


def calc_exit_by_atr(
    df: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    side: str,
    atr_mult_sl: float = 1.2,
    atr_mult_tp: float = 2.0,
) -> Dict[str, float]:
    """
    Exit based on ATR. SL = ATR × X, TP = ATR × Y.
    """
    atr = df.iloc[entry_index]["ATR_14"]

    if side == "long":
        sl = entry_price - atr * atr_mult_sl
        tp = entry_price + atr * atr_mult_tp
    else:
        sl = entry_price + atr * atr_mult_sl
        tp = entry_price - atr * atr_mult_tp

    return {"sl": sl, "tp": tp}


def calc_exit_by_trailing_ema_vwap(
    df: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    side: str,
    trailing_type: str = "EMA_9"
) -> Dict[str, float]:
    """
    Trailing stop method. TP not defined. SL is dynamic: latest EMA or VWAP level.
    """
    trailing_line = df[trailing_type].iloc[-1]
    sl = trailing_line if side == "long" else trailing_line
    return {"sl": sl, "tp": None}  # TP managed by other logic or manual


def calc_exit_on_vwap_loss(df, entry_index, entry_price, side):
    """
    Exit on VWAP break for reclaim strategy.

    Args:
        df (pd.DataFrame): Full OHLCV dataframe with VWAP.
        entry_index (int): Index of trade entry.
        entry_price (float): Entry price.
        side (str): 'long' or 'short'.

    Returns:
        dict: {'sl': ..., 'tp': ...}
    """
    max_lookahead = 40  # Bars to monitor for VWAP break
    vwap_col = "VWAP"

    try:
        for i in range(entry_index + 1, min(entry_index + max_lookahead, len(df))):
            row = df.iloc[i]
            close = row["Close"]
            vwap = row[vwap_col]
            ts = row.name

            if side == "long" and close < vwap:
                log.info(f"[EXIT FOUND] VWAP loss for LONG at index {i} ({ts}): Close={close:.2f} < VWAP={vwap:.2f}")
                return {"sl": close, "tp": None, "exit_index":i}

            elif side == "short" and close > vwap:
                log.info(f"[EXIT FOUND] VWAP loss for SHORT at index {i} ({ts}): Close={close:.2f} > VWAP={vwap:.2f}")
                return {"sl": close, "tp": None, "exit_index":i}

        # === No VWAP loss detected in window ===
        log.info(f"[DEBUG] No VWAP loss exit found for {side.upper()} trade starting at index {entry_index} ({df.index[entry_index]})")
        fallback_exit = df.iloc[min(entry_index + max_lookahead, len(df)-1)]["Close"]
        return {"sl": fallback_exit, "tp": None}

    except Exception as e:
        log.info(f"[ERROR] VWAP loss calculation failed: {e}")
        return {"sl": None, "tp": None}


def calc_exit_on_fade_from_range(
    df: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    side: str,
) -> Dict[str, float]:
    """
    Exit when price fades back inside compression zone + volume drop + reversal bar.
    """
    current = df.iloc[-1]
    inside_range = current["close"] < current["high"] and current["close"] > current["low"]
    vol_fade = current["volume_zscore"] < 0.5
    reversal_bar = current["close"] < current["open"] if side == "long" else current["close"] > current["open"]

    if inside_range and vol_fade and reversal_bar:
        sl = current["close"]  # exit immediately
    else:
        sl = None
    return {"sl": sl, "tp": None}


def calc_exit_on_rejection_trail(
    df: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    side: str,
    rejection_zone_high: float,
    rejection_zone_low: float
) -> Dict[str, float]:
    """
    Exit when price closes back *inside* rejection zone.
    """
    close = df.iloc[-1]["close"]
    if rejection_zone_low <= close <= rejection_zone_high:
        return {"sl": close, "tp": None}  # exit on zone failure
    return {"sl": None, "tp": None}
