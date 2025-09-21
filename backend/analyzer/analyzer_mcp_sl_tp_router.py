from backend.analyzer import analyzer_mcp_sl_tp_logic as logic
from research_agent.logging_setup import get_logger

log = get_logger(__name__)

ROUTER = {
    "sl_tp_fixed_dollar": logic.sl_tp_fixed_dollar,
    "sl_tp_swing_low_high": logic.sl_tp_swing_low_high,
    "sl_tp_dynamic_atr": logic.sl_tp_dynamic_atr,
    "sl_tp_bar_by_bar_trailing": logic.sl_tp_bar_by_bar_trailing,
    "sl_tp_vwap_bands": logic.sl_tp_vwap_bands,
    "sl_tp_custom_zscore": logic.sl_tp_custom_zscore,
    "sl_tp_pivot_level_trailing": logic.sl_tp_pivot_level_trailing,
    "sl_tp_volume_spike": logic.sl_tp_volume_spike,
    "sl_tp_from_r_multiple": logic.sl_tp_from_r_multiple,
    "sl_tp_ema_cross": logic.sl_tp_ema_cross,
    "sl_tp_dmi_bias": logic.sl_tp_dmi_bias,
    "sl_tp_bias_based": logic.sl_tp_bias_based,
    "sl_tp_trailing_update": logic.sl_tp_trailing_update,
}

def dispatch(config):
    fn_name = config.get("sl_tp_function")
    if not fn_name:
        log.info("[SLTP ROUTER] No function name provided in config.")
        return None
    fn = ROUTER.get(fn_name)
    if not fn:
        log.info(f"[SLTP ROUTER] Unknown SL/TP function: {fn_name}")
    return fn
