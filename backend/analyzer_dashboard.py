import numpy as np
import pandas as pd
import plotly.graph_objects as go
import backtrader as bt
class AnalyzerDashboard:
    """
    Visual interface for analyzing strategy performance, trade validity, and classifier signals.
    """

    def __init__(self, df_strategy: pd.DataFrame, df_classifiers: pd.DataFrame):
        """
        Args:
            df_strategy (pd.DataFrame): Strategy dataframe containing OHLC, predictions, etc.
            df_classifiers (pd.DataFrame): Classifier predictions with datetime index.
        """
        self.df_strategy = df_strategy.copy()
        self.df_classifiers = df_classifiers.copy()
        self.plot_df = None  # Used for storing merged/processed view

    def validate_trades_for_plotting(self, trades: list[dict]) -> pd.DataFrame:
        """
        Converts trade list to DataFrame and filters trades with valid entry/exit timestamps.

        Args:
            trades (list[dict]): List of trade records from the strategy.

        Returns:
            pd.DataFrame: Filtered DataFrame with valid trades.
        """
        # ✅ Convert list of trades to DataFrame
        trade_df = pd.DataFrame(trades)

        # ✅ Ensure strategy data has proper datetime index
        self.df_strategy["Datetime"] = pd.to_datetime(
            self.df_strategy["Date"] + " " + self.df_strategy["Time"]
        )
        self.df_strategy.set_index("Datetime", inplace=True)

        # ✅ Rebuild OHLC for alignment check
        ohlc_df = self.df_strategy[["Open", "High", "Low", "Close"]].rename(
            columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
        )

        valid_trades =trade_df[trade_df["entry_time"].isin(self.df_strategy.index)].copy()

        print(f"✅ Valid trades for plotting: {len(valid_trades)} / {len(trade_df)}")
        return valid_trades

    def plot_trades_and_predictions(self, trade_df: pd.DataFrame) -> None:
        """
        Visualize strategy trades, predicted highs, actual highs, and classifier outputs using Plotly.

        Args:
            trade_df (pd.DataFrame): DataFrame with 'entry_time', 'exit_time', 'entry_price', 'exit_price'.

        Returns:
            None – displays an interactive chart.
        """
        import numpy as np

        # Step 1: Build core DataFrame with actual and predicted values
        plot_df = self.df_strategy.copy()
        plot_df["Timestamp"] = pd.to_datetime(plot_df["Date"] + " " + plot_df["Time"])
        plot_df["Actual_High"] = plot_df["High"].shift(-1)
        plot_df = plot_df.dropna(subset=["Actual_High"])
        plot_df.set_index("Timestamp", inplace=True)

        # Step 2: Merge classifier predictions
        plot_df = plot_df.merge(self.df_classifiers, how="left", left_index=True, right_index=True)

        # Step 3: Create vertical offsets for classifier signals
        buffer = (plot_df["High"].max() - plot_df["Low"].min()) * 0.01
        plot_df["rf_y"] = plot_df["Low"] - buffer
        plot_df["lt_y"] = plot_df["Low"] - buffer * 2
        plot_df["xg_y"] = plot_df["Low"] - buffer * 3

        self.plot_df = plot_df  # store for possible reuse

        # Step 4: Start chart
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="Candlestick",
            line=dict(width=1),
            opacity=1.0
        ))

        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df["Predicted"],
            mode='lines+markers',
            name="Predicted High",
            line=dict(color="orange"),
            marker=dict(symbol='x', size=8)
        ))

        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df["Actual_High"],
            mode='lines+markers',
            name="Actual High",
            line=dict(color="blue"),
            marker=dict(symbol='circle', size=6)
        ))

        # Step 5: Plot trade markers
        visible_trades = 0
        for _, trade in trade_df.iterrows():
            entry, exit_, eprice, xprice = trade["entry_time"], trade["exit_time"], trade["entry_price"], trade[
                "exit_price"]
            if entry in plot_df.index and exit_ in plot_df.index:
                visible_trades += 1
                fig.add_trace(go.Scatter(
                    x=[entry],
                    y=[eprice],
                    mode='markers+text',
                    text=[f'Buy @ {eprice:.2f}'],
                    textposition='bottom center',
                    marker=dict(symbol='triangle-up', size=12, color='lime'),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=[exit_],
                    y=[xprice],
                    mode='markers+text',
                    text=[f'Sell @ {xprice:.2f}'],
                    textposition='top center',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    showlegend=False
                ))
                color = 'limegreen' if trade["pnl"] >= 0 else 'red'
                fig.add_trace(go.Scatter(
                    x=[entry, exit_],
                    y=[eprice, xprice],
                    mode='lines',
                    line=dict(color=color, width=3),
                    showlegend=False
                ))

        # Step 6: Classifier signals (if exist)
        for clf, y_col, symbol in [
            ("RandomForest", "rf_y", "diamond"),
            ("LightGBM", "lt_y", "star"),
            ("XGBoost", "xg_y", "cross"),
        ]:
            if clf in plot_df.columns:
                sub = plot_df.dropna(subset=[clf])
                fig.add_trace(go.Scatter(
                    x=sub.index,
                    y=sub[y_col],
                    mode="markers+text",
                    name=f"{clf} (1/0)",
                    marker=dict(
                        symbol=symbol,
                        size=10,
                        color=sub[clf].map({1: "green", 0: "red"})
                    ),
                    showlegend=True
                ))

        print(f"🧮 Trades visible in Plotly chart: {visible_trades} / {len(trade_df)}")

        # Step 7: Layout styling
        fig.update_layout(
            title="📈 Trades + Predictions (with Classifiers)",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=700,
            margin=dict(l=30, r=30, t=40, b=30),
            legend=dict(font=dict(color="white"), bgcolor="black")
        )

        fig.show()

    def plot_equity_curve_with_drawdown(self, df_trades: pd.DataFrame) -> None:
        """
        Plots the equity curve with max drawdown annotation using Plotly.

        Args:
            df_trades (pd.DataFrame): DataFrame with numeric 'pnl' column per trade.

        Returns:
            None – shows the chart in notebook.
        """
        df = df_trades.copy()
        if df["pnl"].dtype == "object":
            df["pnl"] = df["pnl"].replace('[\$,]', '', regex=True).astype(float)

        df["equity"] = df["pnl"].cumsum()
        df["cum_max"] = df["equity"].cummax()
        df["drawdown"] = df["equity"] - df["cum_max"]
        max_dd = df["drawdown"].min()
        end_idx = df["drawdown"].idxmin()
        start_idx = df["equity"][:end_idx].idxmax()

        fig = go.Figure()

        # Equity curve
        fig.add_trace(go.Scatter(
            x=df.index,
            text=df["entry_time"].dt.strftime("%Y-%m-%d %H:%M"),
            hoverinfo='text+y',
            y=df["equity"],
            mode="lines+markers",
            name="Equity Curve",
            line=dict(color="royalblue")
        ))

        # Max drawdown annotation
        fig.add_trace(go.Scatter(
            x=[start_idx, end_idx],
            y=[df.loc[start_idx, "equity"], df.loc[end_idx, "equity"] - 25],
            mode="lines+text",
            line=dict(color="red", dash="dash", width=2),
            name="Max Drawdown",
            text=[None, f"⬇️ Max DD: ${abs(max_dd):.2f}"],
            textposition="top center"
        ))

        fig.update_layout(
            title="📉 Equity Curve with Max Drawdown",
            xaxis_title="Trade Index",
            yaxis_title="Cumulative PnL ($)",
            template="plotly_white",
            height=500,
            showlegend=True
        )

        fig.show()

    def analyze_trade_duration_buckets(self, df_trades: pd.DataFrame) -> None:
        """
        Analyzes how long trades were held in minutes and buckets the distribution.

        Args:
            df_trades (pd.DataFrame): Trades with 'entry_time' and 'exit_time' columns.

        Returns:
            None – prints summary and shows a bar chart.
        """
        df = df_trades.copy()
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"])

        # Calculate duration in minutes
        df["duration_min"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60

        # Bucket durations
        def bucketize(x):
            if x <= 1:
                return "1 min"
            elif x <= 2:
                return "2 min"
            elif x <= 3:
                return "3 min"
            elif x <= 4:
                return "4 min"
            elif x <= 5:
                return "5 min"
            elif x <= 10:
                return "6–10 min"
            else:
                return ">10 min"

        df["duration_bucket"] = df["duration_min"].apply(bucketize)

        # All possible buckets
        ordered_labels = ["1 min", "2 min", "3 min", "4 min", "5 min", "6–10 min", ">10 min"]

        # Count and reindex to force all buckets to appear (even with 0)
        bucket_counts = df["duration_bucket"].value_counts().reindex(ordered_labels, fill_value=0)
        bucket_df = bucket_counts.reset_index()
        bucket_df.columns = ["duration_bucket", "count"]

        # Print summary
        print("📊 Trade Duration Distribution (minutes):")
        print(bucket_df.set_index("duration_bucket")["count"])

        # Plot
        import plotly.express as px
        fig = px.bar(
            bucket_df,
            x="duration_bucket",
            y="count",
            labels={"duration_bucket": "Duration Bucket", "count": "Number of Trades"},
            title="⏱ Trade Duration Distribution",
            template="plotly_white"
        )
        fig.show()

    def calculate_strategy_metrics(self, df_trades: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates key performance metrics from a trade list and returns them as a DataFrame.
        """

        df = df_trades.copy()
        df['pnl'] = df['pnl'].astype(float)
        df['date'] = pd.to_datetime(df['entry_time']).dt.date
        df['duration_min'] = (pd.to_datetime(df['exit_time']) - pd.to_datetime(
            df['entry_time'])).dt.total_seconds() / 60

        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] < 0]
        net_pnl = df['pnl'].sum()
        profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else np.nan
        win_rate = len(wins) / len(df) * 100 if len(df) > 0 else np.nan

        max_consec_losses = max(
            (list(map(len, "".join(['L' if p < 0 else 'W' for p in df['pnl']]).split('W')))), default=0
        )

        outlier_thresh = df['pnl'].std() * 3
        outliers = df[np.abs(df['pnl'] - df['pnl'].mean()) > outlier_thresh]

        metrics = {
            "💰 Overall Performance": {
                "Total Net PnL ($)": net_pnl,
                "Profit Factor": profit_factor,
            },
            "🎯 Trade Quality Metrics": {
                "Win Rate (%)": win_rate,
                "Average Win ($)": wins['pnl'].mean() if not wins.empty else np.nan,
                "Average Loss ($)": losses['pnl'].mean() if not losses.empty else np.nan,
                "Win/Loss Ratio": (wins['pnl'].mean() / abs(losses['pnl'].mean()))
                if not wins.empty and not losses.empty else np.nan,
                "Largest Win ($)": wins['pnl'].max() if not wins.empty else np.nan,
                "Largest Loss ($)": losses['pnl'].min() if not losses.empty else np.nan,
            },
            "📅 Time-Based Metrics": {
                "Avg Daily PnL ($)": df.groupby('date')['pnl'].sum().mean(),
                "Trades per Day": df.groupby('date').size().mean(),
                "Avg Trade Duration (min)": df['duration_min'].mean(),
            },
            "⚠️ Risk / Drawdown Metrics": {
                "Max Drawdown ($)": df['pnl'].cumsum().cummax().sub(df['pnl'].cumsum()).max(),
                "Max Drawdown (%)": 100 * df['pnl'].cumsum().cummax().sub(df['pnl'].cumsum()).div(
                    df['pnl'].cumsum().cummax().replace(0, np.nan)
                ).max(),
                "Max Consecutive Losses": max_consec_losses,
            },
            "📊 Distribution / Reliability": {
                "PnL Std Dev": df['pnl'].std(),
                "Outlier Count (>3σ)": len(outliers),
            }
        }

        # Flatten to single-row DataFrame
        final_rows = {}
        for section, vals in metrics.items():
            for k, v in vals.items():
                final_rows[f"{section} | {k}"] = v

        return pd.DataFrame([final_rows])

    import plotly.graph_objects as go

    def display_metrics_plotly(self, df: pd.DataFrame) -> None:
        import plotly.graph_objects as go

        df = df.transpose().reset_index()
        df.columns = ["Metric", "Value"]

        # Custom formatting for each metric
        formatting = {
            "💰 Overall Performance | Total Net PnL ($)": lambda v: f"${v:,.2f}",
            "💰 Overall Performance | Profit Factor": lambda v: f"{v:.2f}",
            "🎯 Trade Quality Metrics | Win Rate (%)": lambda v: f"{v:.2f}%",
            "🎯 Trade Quality Metrics | Average Win ($)": lambda v: f"${v:.2f}",
            "🎯 Trade Quality Metrics | Average Loss ($)": lambda v: f"${v:.2f}",
            "🎯 Trade Quality Metrics | Win/Loss Ratio": lambda v: f"${v:.2f}",
            "🎯 Trade Quality Metrics | Largest Win ($)": lambda v: f"${v:.2f}",
            "🎯 Trade Quality Metrics | Largest Loss ($)": lambda v: f"${v:.2f}",
            "📅 Time-Based Metrics | Avg Daily PnL ($)": lambda v: f"${v:.2f}",
            "📅 Time-Based Metrics | Trades per Day": lambda v: f"{v:.2f}",
            "📅 Time-Based Metrics | Avg Trade Duration (min)": lambda v: f"{v:.2f} min",
            "⚠️ Risk / Drawdown Metrics | Max Drawdown ($)": lambda v: f"${v:.2f}",
            "⚠️ Risk / Drawdown Metrics | Max Drawdown (%)": lambda v: f"{v:.2f}%",
            "⚠️ Risk / Drawdown Metrics | Max Consecutive Losses": lambda v: f"${v:.2f}",
            "📊 Distribution / Reliability | PnL Std Dev": lambda v: f"${v:.2f}",
            "📊 Distribution / Reliability | Outlier Count (>3σ)": lambda v: f"{v:.2f}"
        }

        df["Value"] = df.apply(lambda row: formatting.get(row["Metric"], lambda v: f"{v:.2f}")(row["Value"]), axis=1)

        fig = go.Figure(data=[go.Table(
            header=dict(values=["📊 Metric", "📈 Value"],
                        fill_color='lightgray',
                        font=dict(size=20, color='black'),
                        align='left'),
            cells=dict(
                values=[df["Metric"], df["Value"]],
                fill_color='white',
                align='left',
                font=dict(size=20),  # Keep your larger font
                height=30  # 🔧 Increase row height to prevent overlap
            )

        )])

        fig.update_layout(width=1400, height=900, margin=dict(t=40, b=40))
        fig.show()

    def display_strategy_and_metrics_side_by_side(self, df_metrics: pd.DataFrame, strategy_params: dict) -> None:
        """
        Display strategy performance metrics and parameters side-by-side using Plotly tables.
        """
        import plotly.graph_objects as go

        # --- Prepare Metrics Table ---
        df_metrics = df_metrics.transpose().reset_index()
        df_metrics.columns = ["Metric", "Value"]

        formatting = {
            "💰 Overall Performance | Total Net PnL ($)": lambda v: f"${v:,.2f}",
            "💰 Overall Performance | Profit Factor": lambda v: f"{v:.2f}",
            "🎯 Trade Quality Metrics | Win Rate (%)": lambda v: f"{v:.2f}%",
            "🎯 Trade Quality Metrics | Average Win ($)": lambda v: f"${v:.2f}",
            "🎯 Trade Quality Metrics | Average Loss ($)": lambda v: f"${v:.2f}",
            "🎯 Trade Quality Metrics | Win/Loss Ratio": lambda v: f"{v:.2f}",
            "🎯 Trade Quality Metrics | Largest Win ($)": lambda v: f"${v:.2f}",
            "🎯 Trade Quality Metrics | Largest Loss ($)": lambda v: f"${v:.2f}",
            "📅 Time-Based Metrics | Avg Daily PnL ($)": lambda v: f"${v:.2f}",
            "📅 Time-Based Metrics | Trades per Day": lambda v: f"{v:.2f}",
            "📅 Time-Based Metrics | Avg Trade Duration (min)": lambda v: f"{v:.2f} min",
            "⚠️ Risk / Drawdown Metrics | Max Drawdown ($)": lambda v: f"${v:.2f}",
            "⚠️ Risk / Drawdown Metrics | Max Drawdown (%)": lambda v: f"{v:.2f}%",
            "⚠️ Risk / Drawdown Metrics | Max Consecutive Losses": lambda v: f"{v:.2f}",
            "📊 Distribution / Reliability | PnL Std Dev": lambda v: f"${v:.2f}",
            "📊 Distribution / Reliability | Outlier Count (>3σ)": lambda v: f"{v:.2f}"
        }

        df_metrics["Value"] = df_metrics.apply(
            lambda row: formatting.get(row["Metric"], lambda v: f"{v:.2f}")(row["Value"]), axis=1
        )


        # --- Prepare Strategy Params Table ---
        df_params = pd.DataFrame(list(strategy_params.items()), columns=["Parameter", "Value"])
        df_params["Value"] = df_params.apply(
            lambda row: f"${row['Value']:.2f}" if isinstance(row["Value"], (int, float)) and "tick" not in row[
                "Parameter"].lower()
            else row["Value"],
            axis=1
        )

        param_formatting = {
            "Strategy Class": lambda v: str(v),
            "Tick Size": lambda v: f"{float(str(v).replace('$', '').replace(',', '')):.2f}",
            "Tick Value ($)": lambda v: f"${float(str(v).replace('$', '').replace(',', '')):.2f}",
            "Contract Size": lambda v: f"{float(str(v).replace('$', '').replace(',', '')):.2f}",
            "Target Ticks": lambda v: f"{int(float(str(v).replace('$', '').replace(',', '')))}",
            "Stop Ticks": lambda v: f"{int(float(str(v).replace('$', '').replace(',', '')))}",
            "Min Distance (pts)": lambda v: f"{float(str(v).replace('$', '').replace(',', '')):.2f} pts",
            "Max Distance (pts)": lambda v: f"{float(str(v).replace('$', '').replace(',', '')):.2f} pts",
            "Min Classifier Signals": lambda v: f"{int(float(str(v).replace('$', '').replace(',', '')))}",
            "Session Start": lambda v: str(v),
            "Session End": lambda v: str(v),
            "Initial Cash ($)": lambda v: f"${float(str(v).replace('$', '').replace(',', '')):,.2f}",
        }

        df_params["Value"] = df_params.apply(
            lambda row: param_formatting.get(row["Parameter"], lambda v: f"{v}")(row["Value"]),
            axis=1
        )

        # --- Create Plotly Subplots ---
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["📊 Strategy Metrics", "🧠 Strategy Parameters"],
            specs=[[{"type": "table"}, {"type": "table"}]]
        )

        fig.add_trace(go.Table(
            header=dict(values=["📊 Metric", "📈 Value"],
                        fill_color='lightgray',
                        font=dict(size=18, color='black'),
                        align='left'),
            cells=dict(
                values=[df_metrics["Metric"], df_metrics["Value"]],
                fill_color='white',
                align='left',
                font=dict(size=18),
                height=30
            )
        ), row=1, col=1)

        fig.add_trace(go.Table(
            header=dict(values=["⚙️ Parameter", "🔢 Value"],
                        fill_color='lightgray',
                        font=dict(size=18, color='black'),
                        align='left'),
            cells=dict(
                values=[df_params["Parameter"], df_params["Value"]],
                fill_color='white',
                align='left',
                font=dict(size=18),
                height=30
            )
        ), row=1, col=2)

        fig.update_layout(
            width=1600,
            height=900,
            margin=dict(t=40, b=40)
        )

        fig.show()

    def plot_trades_and_predictions_intrabar_1_min(self, trade_df: pd.DataFrame, df_1min: pd.DataFrame) -> None:
        """
        Visualize intrabar trades and predictions using 1-min bars, highlighting 5-min bars with filled candles.

        Args:
            trade_df (pd.DataFrame): Trade records with entry/exit times and prices.
            df_1min (pd.DataFrame): 1-minute OHLC data with datetime index.

        Returns:
            None – displays interactive chart.
        """
        import plotly.graph_objects as go

        # Ensure datetime index
        df = df_1min.copy()
        df.index = pd.to_datetime(df.index)

        # Detect 5-min anchor bars (filled) vs. 1-min (hollow)
        df["is_5min"] = df.index.to_series().dt.minute % 5 == 0
        df["bar_color"] = df.apply(
            lambda row: "rgba(0,255,0,0.6)" if row["Open"] < row["Close"] else "rgba(255,0,0,0.6)",
            axis=1
        )
        df["line_color"] = df.apply(
            lambda row: "green" if row["Open"] < row["Close"] else "red",
            axis=1
        )

        # Start chart
        fig = go.Figure()

        # 1-min bars (all)
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color="green",
            decreasing_line_color="red",
            increasing_fillcolor="rgba(0,0,0,0)",  # Hollow
            decreasing_fillcolor="rgba(0,0,0,0)",  # Hollow
            name="1-min Bars"
        ))

        # Overlay filled 5-min bars
        df_5min = df[df["is_5min"]]
        fig.add_trace(go.Bar(
            x=df_5min.index,
            y=df_5min["Close"],
            width=60 * 1000,  # 1 minute in ms
            marker_color=df_5min["bar_color"],
            opacity=0.3,
            name="5-min Fills",
            hoverinfo="skip"
        ))


        # Add predicted and actual highs
        if "Predicted" in df.columns:

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["Predicted"],
                mode='lines+markers',
                name="Predicted High",
                line=dict(color="orange"),
                marker=dict(symbol='x', size=8),
                connectgaps=True
            ))

        if "Next_High" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["Next_High"],
                mode='lines+markers',
                name="Actual High",
                line=dict(color="skyblue", dash='dash'),
                marker=dict(symbol='circle', size=6),
                connectgaps=True
            ))

        # Add trades (lines + markers)
        visible_trades = 0
        for _, row in trade_df.iterrows():
            entry_time = pd.to_datetime(row["entry_time"])
            exit_time = pd.to_datetime(row["exit_time"])
            entry_price = row["entry_price"]
            exit_price = row["exit_price"]
            pnl = row["pnl"]
            color = "limegreen" if pnl >= 0 else "red"

            # Line
            fig.add_trace(go.Scatter(
                x=[entry_time, exit_time],
                y=[entry_price, exit_price],
                mode="lines",
                line=dict(color=color, width=3),
                showlegend=False
            ))

            # Entry marker
            fig.add_trace(go.Scatter(
                x=[entry_time],
                y=[entry_price],
                mode="markers+text",
                text=[f'Buy @ {entry_price:.2f}'],
                textposition="bottom center",
                marker=dict(symbol="triangle-up", color="lime", size=12),
                showlegend=False
            ))

            # Exit marker
            fig.add_trace(go.Scatter(
                x=[exit_time],
                y=[exit_price],
                mode="markers+text",
                text=[f'Sell @ {exit_price:.2f}'],
                textposition="top center",
                marker=dict(symbol="triangle-down", color="red", size=12),
                showlegend=False
            ))
            visible_trades += 1

        print(f"🧮 Trades plotted on chart: {visible_trades} / {len(trade_df)}")
        y_min = min(df_1min["Low"].min(), trade_df["entry_price"].min(), trade_df["exit_price"].min())
        y_max = max(df_1min["High"].max(), trade_df["entry_price"].max(), trade_df["exit_price"].max())


        # Layout
        fig.update_layout(
            title="📈 Intrabar Strategy — 1-min Chart (with 5-min Highlights)",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            yaxis=dict(range=[y_min * 0.98, y_max * 1.02]),
            height=750,
            margin=dict(t=40, b=30)
        )

        fig.show()

    def build_trade_dataframe_from_orders(self, order_list: list) -> pd.DataFrame:
        """
        Construct a clean trade DataFrame from filled BUY and SELL orders after strategy run.

        Args:
            order_list (List[bt.Order]): List of all broker orders.

        Returns:
            pd.DataFrame: Clean trade log with entry/exit times, prices, PnL, duration.
        """
        orders_df = pd.DataFrame([{
            'ref': o.ref,
            'type': 'SELL' if o.issell() else 'BUY',
            'status': o.getstatusname(),
            'exec_type': o.exectype,
            'submitted_price': o.created.price,
            'filled_price': o.executed.price if o.status == bt.Order.Completed else None,
            'executed_dt': bt.num2date(o.executed.dt) if o.status == bt.Order.Completed else None
        } for o in order_list])

        # Filter only completed and sort
        orders_df = orders_df[orders_df['status'] == 'Completed'].copy()
        orders_df.sort_values(by='executed_dt', inplace=True)

        trades = []
        buy_stack = []

        for _, row in orders_df.iterrows():
            if row['type'] == 'BUY':
                buy_stack.append(row)
            elif row['type'] == 'SELL' and buy_stack:
                buy = buy_stack.pop(0)
                sell = row
                entry_time = buy['executed_dt']
                exit_time = sell['executed_dt']
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': buy['filled_price'],
                    'exit_time': exit_time,
                    'exit_price': sell['filled_price'],
                    'pnl': sell['filled_price'] - buy['filled_price'],
                    'duration_min': (exit_time - entry_time).total_seconds() / 60
                })

        return pd.DataFrame(trades)



