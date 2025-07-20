# === Static SL/TP Strategies ===
import pandas as pd
from typing import Dict



def sl_tp_fixed_dollar(
    df: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    side: str,
    stop_dollar: float,
    tp_dollar: float
) -> Dict[str, float]:
    """
    SL/TP logic with fixed dollar distance:
    - SL = entry_price - stop_dollar (or + for short)
    - TP = entry_price + tp_dollar (or - for short)

    Parameters:
        df: OHLC dataframe
        entry_index: Index of entry candle
        entry_price: Price of entry
        side: 'long' or 'short'
        stop_dollar: Dollar distance to stop
        tp_dollar: Dollar distance to take profit

    Returns:
        Dict with keys 'sl' and 'tp'
    """
    pass

def sl_tp_swing_low_high(
    df: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    side: str,
    lookback: int,
    r_multiple: float
) -> Dict[str, float]:
    """
    SL/TP logic using recent swing low/high for stop, R-multiple for TP:
    - SL = recent swing low (long) or high (short) in lookback window
    - TP = entry_price + (risk × r_multiple)

    Parameters:
        df: OHLC dataframe
        entry_index: Index of entry candle
        entry_price: Price of entry
        side: 'long' or 'short'
        lookback: Number of bars to look back for swing low/high
        r_multiple: TP multiplier (e.g., 2.0 for 2R)

    Returns:
        Dict with keys 'sl' and 'tp'
    """
    pass

def sl_tp_dynamic_atr(
    df: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    side: str,
    atr_period: int,
    atr_mult_sl: float,
    atr_mult_tp: float
) -> Dict[str, float]:
    """
    SL/TP logic using ATR for dynamic stop and take profit:
    - SL = entry_price - (ATR × atr_mult_sl) (or + for short)
    - TP = entry_price + (ATR × atr_mult_tp) (or - for short)

    Parameters:
        df: OHLC dataframe
        entry_index: Index of entry candle
        entry_price: Price of entry
        side: 'long' or 'short'
        atr_period: ATR lookback period
        atr_mult_sl: Multiplier for stop loss
        atr_mult_tp: Multiplier for take profit

    Returns:
        Dict with keys 'sl' and 'tp'
    """
    pass

def sl_tp_bar_by_bar_trailing(
    df: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    side: str,
    trail_lookback: int
) -> Dict[str, float]:
    """
    Bar-by-bar trailing stop logic:
    - SL = lowest low (long) or highest high (short) of previous trail_lookback bars
    - TP = not defined (set to None or entry_price for placeholder)

    Parameters:
        df: OHLC dataframe
        entry_index: Index of entry candle
        entry_price: Price of entry
        side: 'long' or 'short'
        trail_lookback: Number of bars to look back for trailing stop

    Returns:
        Dict with keys 'sl' and 'tp'
    """
    pass

def sl_tp_vwap_bands(
    df: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    side: str,
    vwap_std_mult: float
) -> Dict[str, float]:
    """
    VWAP bands SL/TP logic:
    - SL = VWAP - vwap_std_mult × std (long) or + for short
    - TP = VWAP + vwap_std_mult × std (long) or - for short

    Parameters:
        df: OHLC dataframe
        entry_index: Index of entry candle
        entry_price: Price of entry
        side: 'long' or 'short'
        vwap_std_mult: Multiplier for VWAP standard deviation band

    Returns:
        Dict with keys 'sl' and 'tp'
    """
    pass

def sl_tp_from_r_multiple(df, entry_index, entry_price, side, tick_size, stop_ticks, r_multiple):
    """
    SL/TP logic based on R-multiple.

    Parameters:
        df: OHLC dataframe
        entry_index: Index of entry bar
        entry_price: Price at which trade was entered
        side: 'long' or 'short'
        tick_size: Min price movement (e.g. 0.25)
        stop_ticks: Distance from entry to stop in ticks
        r_multiple: Reward-to-risk multiplier for TP (e.g. 2.0 for 2R)

    Returns:
        dict: {"sl": stop_loss_price, "tp": take_profit_price}
    """
    risk = stop_ticks * tick_size
    if side == "long":
        sl = entry_price - risk
        tp = entry_price + (r_multiple * risk)
    else:
        sl = entry_price + risk
        tp = entry_price - (r_multiple * risk)
    return {"sl": round(sl, 6), "tp": round(tp, 6)}
