import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

        # ✅ Filter trades that exist in the OHLC datetime index
        valid_trades = trade_df[
            trade_df["entry_time"].isin(ohlc_df.index) &
            trade_df["exit_time"].isin(ohlc_df.index)
            ].copy()

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


