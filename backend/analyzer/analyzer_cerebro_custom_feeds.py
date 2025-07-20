

import backtrader as bt

class CustomRegressionData(bt.feeds.PandasData):
    """
    Backtrader data feed for regression strategies with predicted_high and predicted_low columns.
    Supports standard OHLCV fields plus two custom lines.
    """
    lines = ('predicted_high', 'predicted_low')
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1),
        ('predicted_high', 'predicted_high'),
        ('predicted_low', 'predicted_low'),
    )