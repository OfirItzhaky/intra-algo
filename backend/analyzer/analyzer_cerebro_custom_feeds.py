

import backtrader as bt

class CustomClassifierData(bt.feeds.PandasData):
    """
    Custom data feed for Backtrader that includes classifier columns and predicted high.
    """
    lines = ('predicted_high', 'RandomForest', 'LightGBM', 'XGBoost', 'multi_class_label')
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('openinterest', -1),
        ('predicted_high', -1),
        ('RandomForest', -1),
        ('LightGBM', -1),
        ('XGBoost', -1),
        ('multi_class_label', -1),
    )


class CustomRegressionData(bt.feeds.PandasData):
    lines = ('predicted_high', 'predicted_low',)
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
        ('predicted_high', 'predicted_high'),
        ('predicted_low', 'predicted_low'),
    )


class CustomVWAPData(bt.feeds.PandasData):
    """
    Custom Backtrader data feed to expose all VWAP-related indicators used across all strategies.
    """
    lines = (
        'VWAP', 'EMA_9', 'EMA_20', 'ATR_14',
        'VWAP_upper', 'VWAP_lower',
        'DMP_14', 'DMN_14', 'ADX_14',
        'volume_zscore', 'ema_bias_filter', 'dmi_crossover'
    )

    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'vol'),
        ('openinterest', -1),
        ('VWAP', 'VWAP'),
        ('EMA_9', 'EMA_9'),
        ('EMA_20', 'EMA_20'),
        ('ATR_14', 'ATR_14'),
        ('VWAP_upper', 'VWAP_upper'),
        ('VWAP_lower', 'VWAP_lower'),
        ('DMP_14', 'DMP_14'),
        ('DMN_14', 'DMN_14'),
        ('ADX_14', 'ADX_14'),
        ('volume_zscore', 'volume_zscore'),
        ('ema_bias_filter', 'ema_bias_filter'),
        ('dmi_crossover', 'dmi_crossover'),
    )

