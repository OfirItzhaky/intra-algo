import backtrader as bt
import pandas as pd
from backend.analyzer_strategy_blueprint import ElasticNetStrategy
from backend.analyzer_cerebro_custom_feeds import CustomRegressionData

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
        session_end: str,
        max_daily_profit: float = 36.0,
        max_daily_loss: float = -36.0
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
        self.max_daily_profit = max_daily_profit
        self.max_daily_loss = max_daily_loss

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
        Runs Long5min1minStrategy with daily PnL limits
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
            multi_class_threshold=multi_class_threshold,
            max_daily_profit=self.max_daily_profit,
            max_daily_loss=self.max_daily_loss
        )

        data_5min = CustomClassifierData(dataname=df_5min)
        data_1min = CustomClassifierData(dataname=df_1min)

        cerebro.adddata(data_1min)  # Intrabar data feed
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        print("🚀 Running Long5min1minStrategy backtest...")
        results = cerebro.run()
        return results, cerebro

    def run_backtest_RegressionScalpingStrategy(
        self,
        df_5min: pd.DataFrame,
        df_1min: pd.DataFrame,
        params: dict,
        config_index: int = None,
        total_configs: int = None,
        long_t: float = None,
        short_t: float = None,
        min_vol: float = None,
        bar_color: bool = None
    ) -> tuple:
        """
        Runs RegressionScalpingStrategy with CustomRegressionData (5m) and standard PandasData (1m).
        params: dict of strategy parameters to pass to RegressionScalpingStrategy.
        Returns (results, strategy_instance, cerebro)
        """
        from backend.analyzer_strategy_blueprint import RegressionScalpingStrategy
        cerebro = bt.Cerebro()
        # Prepare 5min data feed
        df_5min = df_5min.copy()
        # --- Always use lowercase columns ---
        df_5min.columns = [col.lower() for col in df_5min.columns]
        if 'datetime' not in df_5min.columns:
            if 'date' in df_5min.columns and 'time' in df_5min.columns:
                df_5min['datetime'] = pd.to_datetime(df_5min['date'] + ' ' + df_5min['time'])
            else:
                raise KeyError(f"'date' and/or 'time' columns not found in df_5min. Columns: {list(df_5min.columns)}")
        df_5min.set_index('datetime', inplace=True)
        # Ensure predicted_high and predicted_low columns exist
        if 'predicted_high' not in df_5min.columns and 'predicted' in df_5min.columns:
            df_5min['predicted_high'] = df_5min['predicted']
        if 'predicted_low' not in df_5min.columns:
            df_5min['predicted_low'] = float('nan')
        df_5min.index = pd.to_datetime(df_5min.index)
        # CustomRegressionData feed expects lowercase columns
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
        data_5min = CustomRegressionData(dataname=df_5min)
        # Prepare 1min data feed
        df_1min = df_1min.copy()
        # 🔧 Normalize columns to lowercase
        df_1min.columns = [col.lower() for col in df_1min.columns]
        # Dynamically detect the volume column
        if 'volume' in df_1min.columns:
            vol_col = 'volume'
        else:
            vol_col = next((col for col in df_1min.columns if 'vol' in col), None)
        if not vol_col:
            raise ValueError("No volume column found in 1-min data. Tried to match 'volume' or any column containing 'vol'. Columns: " + str(list(df_1min.columns)))
        vol_idx = df_1min.columns.get_loc(vol_col)
        # Define custom 1-min data feed class with correct volume mapping by index
        class CustomPandas1Min(bt.feeds.PandasData):
            lines = (vol_col,)
            params = ((vol_col, vol_idx),)
        if 'datetime' not in df_1min.columns:
            df_1min['datetime'] = pd.to_datetime(df_1min['date'] + ' ' + df_1min['time'])
        df_1min.set_index('datetime', inplace=True)
        df_1min.index = pd.to_datetime(df_1min.index)
        data_1min = CustomPandas1Min(dataname=df_1min)
        # Add feeds
        cerebro.adddata(data_5min)  # data0: 5m
        cerebro.adddata(data_1min)  # data1: 1m
        # Add strategy
        print(f"[DEBUG] Params sent to RegressionScalpingStrategy: {params}")
        cerebro.addstrategy(RegressionScalpingStrategy, **params)
        # Add analyzer
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        # Memory usage before cerebro.run()
        print("[DEBUG] Attempting to import psutil for memory logging", flush=True)
        try:
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
        except ImportError:
            psutil = None
            mem_before = None
            print("[DEBUG] psutil not available, skipping memory logging", flush=True)
            print("[ERROR] psutil is required for memory logging. Please install it with: pip install psutil", flush=True)
        # Run
        if config_index is not None and total_configs is not None:
            print(f"🚀 Running RegressionScalpingStrategy backtest {config_index+1}/{total_configs} (long={long_t}, short={short_t}, min_vol={min_vol}, bar_color={bar_color}) ...")
        else:
            print("🚀 Running RegressionScalpingStrategy backtest (CustomRegressionData + PandasData)...")
        results = cerebro.run()
        # Memory usage after cerebro.run()
        if psutil:
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_delta = mem_after - mem_before if mem_before is not None else 0
            print(f"[Memory] Config {config_index+1 if config_index is not None else '?'} — long={long_t}, short={short_t}, min_vol={min_vol}, bar_color={bar_color}: {mem_before:.1f} MB -> {mem_after:.1f} MB (delta: {mem_delta:+.1f} MB)", flush=True)
        # Get the strategy instance (first in results)
        strategy_instance = results[0] if results else None
        # Print a single summary line per iteration
        trades_attr = getattr(strategy_instance, 'trades', None)
        if isinstance(trades_attr, (list, tuple)):
            num_trades = len(trades_attr)
        else:
            num_trades = 0
        print(f"✅ Finished RegressionScalpingStrategy backtest {config_index+1}/{total_configs} (long={long_t}, short={short_t}, min_vol={min_vol}, bar_color={bar_color}) | Trades: {num_trades}")
        # Always return a dict as the first value for downstream code
        if isinstance(results, list) and (not results or not isinstance(results[0], dict)):
            strategy_instance = results[0] if results else None
            sim = {}
            if strategy_instance is not None:
                # Extract serializable fields if present
                trades = getattr(strategy_instance, 'trades', None)
                if trades is not None:
                    sim['trades'] = trades
                backtest_summary = getattr(strategy_instance, 'backtest_summary', None)
                if backtest_summary is not None:
                    sim['backtest_summary'] = backtest_summary
            sim['results'] = results
        else:
            sim = results[0] if results else {}
        return sim, strategy_instance, cerebro
