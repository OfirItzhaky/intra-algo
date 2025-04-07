import backtrader as bt
import pandas as pd
from analyzer_strategy_blueprint import ElasticNetStrategy  # Make sure it's correct path

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

        # Make sure multi_class_label is available, even if None
        if "multi_class_label" not in self.df_strategy.columns:
            self.df_strategy["multi_class_label"] = None
        
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



    def run_backtest_Long5min1minStrategy(self, df_5min: pd.DataFrame, df_1min: pd.DataFrame, strategy_class, use_multi_class=False, multi_class_threshold=3) -> tuple:
        """
        Runs Long5min1minStrategy using fresh Cerebro setup with both 5-min and 1-min data.

        Parameters:
            df_5min (pd.DataFrame): 5-minute strategy DataFrame with predictions and classifiers.
            df_1min (pd.DataFrame): 1-minute OHLC data.
            strategy_class: The Backtrader strategy class to use (e.g., Long5min1minStrategy).
            use_multi_class (bool): Whether to use multi-class labels instead of binary.
            multi_class_threshold (int): Threshold for considering multi-class labels as positive.

        Returns:
            tuple: (results, cerebro) from Backtrader engine.
        """
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(
            commission=0.0,
            mult=self.contract_size * self.tick_value / self.tick_size
        )

        df_5min = df_5min.copy()
        df_5min["datetime"] = pd.to_datetime(df_5min["Date"] + " " + df_5min["Time"])
        df_5min.set_index("datetime", inplace=True)
        df_5min["predicted_high"] = df_5min["Predicted"]

        df_5min.index = pd.to_datetime(df_5min.index)
        df_1min.index = pd.to_datetime(df_1min.index)
        df_1min = df_1min.asfreq('T')  # 'T' = 1 minute
        df_1min = df_1min.rename(columns={"Predicted": "predicted_high"})

        cerebro.addstrategy(
            strategy_class,
            tick_size=self.tick_size,
            tick_value=self.tick_value,
            contract_size=self.contract_size,
            target_ticks=self.target_ticks,
            stop_ticks=self.stop_ticks,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            min_classifier_signals=self.min_classifier_signals,
            session_start=self.session_start,
            session_end=self.session_end,
            use_multi_class=use_multi_class,
            multi_class_threshold=multi_class_threshold
        )

        data_5min = CustomClassifierData(dataname=df_5min)
        data_1min = CustomClassifierData(dataname=df_1min)
        print(data_1min.lines.getlinealiases())  # ⬅️ BREAK HERE

        # cerebro.adddata(data_5min)  # Main data feed
        cerebro.adddata(data_1min)  # Intrabar data feed

        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        print("🚀 Running Long5min1minStrategy backtest...")
        results = cerebro.run()
        return results, cerebro
