import backtrader as bt
import pandas as pd
from analyzer_strategy_blueprint import ElasticNetStrategy  # Make sure it's correct path

class CustomClassifierData(bt.feeds.PandasData):
    """
    Custom data feed for Backtrader that includes classifier columns and predicted high.
    """
    lines = ('predicted_high', 'RandomForest', 'LightGBM', 'XGBoost')
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
    )


class CerebroStrategyEngine:
    """
    Engine to run trading strategies via Backtrader using classifier-enhanced data.
    """

    def __init__(
        self,
        df_strategy: pd.DataFrame,
        df_classifiers: pd.DataFrame,
        initial_cash: float,
        tick_size: float,
        tick_value: float,
        contract_size: int,
        target_ticks: int,
        stop_ticks: int,
        min_dist: float,
        max_dist: float,
        min_classifier_signals: int,
        session_start: str,
        session_end: str
    ):
        """
        Initialize the engine with strategy data, classifier predictions, and all required strategy parameters.
        """
        self.df_strategy = df_strategy.copy()
        self.df_classifiers = df_classifiers.copy()
        self.initial_cash = initial_cash
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.contract_size = contract_size
        self.target_ticks = target_ticks
        self.stop_ticks = stop_ticks
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.min_classifier_signals = min_classifier_signals
        self.session_start = session_start
        self.session_end = session_end

        self.results = None
        self.cerebro = bt.Cerebro()


    def prepare_data(self) -> pd.DataFrame:
        """
        Merges classifier predictions and formats datetime index for Backtrader.
        """
        self.df_strategy["datetime"] = pd.to_datetime(self.df_strategy["Date"] + " " + self.df_strategy["Time"])
        self.df_strategy.set_index("datetime", inplace=True)

        self.df_classifiers.index = pd.to_datetime(self.df_classifiers.index)
        self.df_strategy = self.df_strategy.merge(self.df_classifiers, how="left", left_index=True, right_index=True)
        self.df_strategy["predicted_high"] = self.df_strategy["Predicted"]

        if ElasticNetStrategy.params.min_classifier_signals > 0:
            start_idx = self.df_classifiers.index.min()
            print(f"⏳ Classifier signals begin at: {start_idx}")
            self.df_strategy = self.df_strategy[self.df_strategy.index >= start_idx]

        return self.df_strategy

    def run_backtest(self) -> list:
        """
        Adds strategy and data to Cerebro, sets broker settings, and runs the backtest.
        """
        df_prepared = self.prepare_data()
        data_feed = CustomClassifierData(dataname=df_prepared)

        self.cerebro.addstrategy(
            ElasticNetStrategy,
            tick_size=self.tick_size,
            tick_value=self.tick_value,
            contract_size=self.contract_size,
            target_ticks=self.target_ticks,
            stop_ticks=self.stop_ticks,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            min_classifier_signals=self.min_classifier_signals,
            session_start=self.session_start,
            session_end=self.session_end
        )

        self.cerebro.adddata(data_feed)
        self.cerebro.broker.setcash(self.initial_cash)
        self.cerebro.broker.setcommission(commission=0.0, mult=1 * 1.25 / 0.25)

        print("🚀 Running backtest...")
        self.results = self.cerebro.run()
        return self.results
