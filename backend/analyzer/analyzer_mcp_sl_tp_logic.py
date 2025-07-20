# === Static SL/TP Strategies ===
import pandas as pd
from typing import Dict
import numpy as np



def sl_tp_fixed_dollar(df, entry_index, entry_price, side, stop_dollar, tp_dollar):
    """
    SL/TP logic based on fixed dollar values.

    Parameters:
        df: OHLC dataframe
        entry_index: Index of entry bar
        entry_price: Trade entry price
        side: 'long' or 'short'
        stop_dollar: Dollar distance to stop loss
        tp_dollar: Dollar distance to take profit

    Returns:
        dict: {"sl": stop_loss_price, "tp": take_profit_price}
    """
    if side == "long":
        sl = entry_price - stop_dollar
        tp = entry_price + tp_dollar
    else:
        sl = entry_price + stop_dollar
        tp = entry_price - tp_dollar
    return {"sl": round(sl, 6), "tp": round(tp, 6)}


def sl_tp_swing_low_high(df, entry_index, entry_price, side, lookback, r_multiple):
    """
    SL = swing low (long) or swing high (short), TP = R-multiple from entry.

    Parameters:
        df: OHLC data
        entry_index: Entry bar index
        entry_price: Entry price
        side: 'long' or 'short'
        lookback: Number of bars to look back for swing level
        r_multiple: Multiplier for TP relative to risk

    Returns:
        dict: {"sl": ..., "tp": ...}
    """
    if side == "long":
        swing_low = df["low"].iloc[entry_index - lookback:entry_index].min()
        risk = entry_price - swing_low
        tp = entry_price + r_multiple * risk
        sl = swing_low
    else:
        swing_high = df["high"].iloc[entry_index - lookback:entry_index].max()
        risk = swing_high - entry_price
        tp = entry_price - r_multiple * risk
        sl = swing_high
    return {"sl": round(sl, 6), "tp": round(tp, 6)}


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
    SL/TP logic using ATR for dynamic stop and take profit.
    - SL = entry_price - (ATR × atr_mult_sl) (long) or + (short)
    - TP = entry_price + (ATR × atr_mult_tp) (long) or - (short)

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
    high = df['high']
    low = df['low']
    close = df['close']
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = tr.rolling(window=atr_period, min_periods=1).mean()
    atr_val = float(atr.iloc[entry_index])
    if side == 'long':
        sl = entry_price - atr_val * atr_mult_sl
        tp = entry_price + atr_val * atr_mult_tp
    else:
        sl = entry_price + atr_val * atr_mult_sl
        tp = entry_price - atr_val * atr_mult_tp
    return {"sl": round(sl, 6), "tp": round(tp, 6)}

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
    - TP = None

    Parameters:
        df: OHLC dataframe
        entry_index: Index of entry candle
        entry_price: Price of entry
        side: 'long' or 'short'
        trail_lookback: Number of bars to look back for trailing stop

    Returns:
        Dict with keys 'sl' and 'tp' (tp is None)
    """
    if side == 'long':
        sl = df['low'].iloc[max(0, entry_index - trail_lookback + 1):entry_index + 1].min()
    else:
        sl = df['high'].iloc[max(0, entry_index - trail_lookback + 1):entry_index + 1].max()
    return {"sl": round(sl, 6), "tp": None}

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
    closes = df['close'].iloc[:entry_index + 1]
    volumes = df['vol'].iloc[:entry_index + 1]
    vwap = (closes * volumes).sum() / volumes.sum()
    std = closes.std()
    if side == 'long':
        sl = vwap - vwap_std_mult * std
        tp = vwap + vwap_std_mult * std
    else:
        sl = vwap + vwap_std_mult * std
        tp = vwap - vwap_std_mult * std
    return {"sl": round(sl, 6), "tp": round(tp, 6)}

def sl_tp_custom_zscore(
    df: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    side: str,
    z_threshold: float,
    mean: float,
    std_dev: float
) -> Dict[str, float]:
    """
    Z-score based SL/TP logic:
    - SL = mean - z_threshold × std_dev (long) or + for short
    - TP = mean + z_threshold × std_dev (long) or - for short

    Parameters:
        df: OHLC dataframe
        entry_index: Index of entry candle
        entry_price: Price of entry
        side: 'long' or 'short'
        z_threshold: Z-score threshold
        mean: Mean value for calculation
        std_dev: Standard deviation for calculation

    Returns:
        Dict with keys 'sl' and 'tp'
    """
    if side == 'long':
        sl = mean - z_threshold * std_dev
        tp = mean + z_threshold * std_dev
    else:
        sl = mean + z_threshold * std_dev
        tp = mean - z_threshold * std_dev
    return {"sl": round(sl, 6), "tp": round(tp, 6)}

def sl_tp_pivot_level_trailing(
    df: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    side: str,
    pivot_level: float,
    trailing_offset: float
) -> Dict[str, float]:
    """
    Pivot level trailing SL/TP logic:
    - SL = pivot_level - trailing_offset (long) or + for short
    - TP = pivot_level + trailing_offset (long) or - for short

    Parameters:
        df: OHLC dataframe
        entry_index: Index of entry candle
        entry_price: Price of entry
        side: 'long' or 'short'
        pivot_level: Pivot price level
        trailing_offset: Offset from pivot for SL/TP

    Returns:
        Dict with keys 'sl' and 'tp'
    """
    if side == 'long':
        sl = pivot_level - trailing_offset
        tp = pivot_level + trailing_offset
    else:
        sl = pivot_level + trailing_offset
        tp = pivot_level - trailing_offset
    return {"sl": round(sl, 6), "tp": round(tp, 6)}

def sl_tp_volume_spike(
    df: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    side: str,
    vol_threshold: float,
    drop_pct: float
) -> Dict[str, float]:
    """
    Volume spike SL/TP logic:
    - SL is triggered if volume below vol_threshold or drops by drop_pct vs. previous bar
    - TP = entry_price + 2.0 (long) or -2.0 (short)

    Parameters:
        df: OHLC dataframe
        entry_index: Index of entry candle
        entry_price: Price of entry
        side: 'long' or 'short'
        vol_threshold: Volume threshold for SL
        drop_pct: Percentage drop threshold (0.05 = 5%)

    Returns:
        Dict with keys 'sl' and 'tp'
    """
    vol = df['vol'].iloc[entry_index]
    prev_vol = df['vol'].iloc[entry_index - 1] if entry_index > 0 else vol
    sl_triggered = vol < vol_threshold or (prev_vol > 0 and (vol / prev_vol) < (1 - drop_pct))
    if side == 'long':
        tp = entry_price + 2.0
        sl = entry_price - 2.0 if sl_triggered else None
    else:
        tp = entry_price - 2.0
        sl = entry_price + 2.0 if sl_triggered else None
    return {"sl": round(sl, 6) if sl is not None else None, "tp": round(tp, 6)}

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

def sl_tp_ema_cross(df, entry_index, entry_price, side, fast_period, slow_period):
    """
    SL/TP logic based on EMA cross zones.

    Parameters:
        df: OHLC data
        entry_index: Entry bar
        entry_price: Entry price
        side: 'long' or 'short'
        fast_period: Fast EMA period
        slow_period: Slow EMA period

    Returns:
        dict: {"sl": ..., "tp": ...}
    """
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast_period).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_period).mean()

    if side == "long":
        sl = df["ema_slow"].iloc[entry_index]
        tp = df["ema_fast"].iloc[entry_index]
    else:
        sl = df["ema_fast"].iloc[entry_index]
        tp = df["ema_slow"].iloc[entry_index]
    return {"sl": round(sl, 6), "tp": round(tp, 6)}

def sl_tp_dmi_bias(df, entry_index, entry_price, side, dmi_bias):
    """
    SL/TP zones adjusted based on DMI strength (bias as multiplier).

    Parameters:
        df: OHLC data
        entry_index: Entry bar
        entry_price: Entry price
        side: 'long' or 'short'
        dmi_bias: Strength or conviction signal, scales risk

    Returns:
        dict: {"sl": ..., "tp": ...}
    """
    base_risk = 1.0  # hardcoded for now; can be tuned
    if side == "long":
        sl = entry_price - dmi_bias * base_risk
        tp = entry_price + dmi_bias * 2 * base_risk
    else:
        sl = entry_price + abs(dmi_bias) * base_risk
        tp = entry_price - abs(dmi_bias) * 2 * base_risk
    return {"sl": round(sl, 6), "tp": round(tp, 6)}

def sl_tp_bias_based(df, entry_index, entry_price, side, bias_strength):
    """
    Macro or LLM-detected bias strength used to widen/narrow SL/TP zones.

    Parameters:
        df: OHLC
        entry_index: Entry index
        entry_price: Entry price
        side: 'long' or 'short'
        bias_strength: Range [0.0, 1.0], affects buffer size

    Returns:
        dict: {"sl": ..., "tp": ...}
    """
    base_buffer = 1.0
    if side == "long":
        sl = entry_price - base_buffer * (1 + bias_strength)
        tp = entry_price + base_buffer * (1 + bias_strength * 2)
    else:
        sl = entry_price + base_buffer * (1 + bias_strength)
        tp = entry_price - base_buffer * (1 + bias_strength * 2)
    return {"sl": round(sl, 6), "tp": round(tp, 6)}

def sl_tp_trailing_update(
    df: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    side: str,
    trail_lookback: int
) -> Dict[str, float]:
    """
    Bar-by-bar SL update logic that tightens the stop after entry.
    SL moves to the most recent N-bar low (long) or high (short) up to entry_index.
    TP is always None (trailing stop only).

    Parameters:
        df: OHLC dataframe
        entry_index: Index of the current candle
        entry_price: Price of entry
        side: 'long' or 'short'
        trail_lookback: Number of bars to look back for trailing stop

    Returns:
        Dict with keys 'sl' (stop loss) and 'tp' (always None)
    """
    if side == 'long':
        sl = df['low'].iloc[max(0, entry_index - trail_lookback + 1):entry_index + 1].min()
    else:
        sl = df['high'].iloc[max(0, entry_index - trail_lookback + 1):entry_index + 1].max()
    return {"sl": round(sl, 6), "tp": None}
