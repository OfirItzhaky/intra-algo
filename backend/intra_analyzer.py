#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # ✅ Clean up __pycache__ and .ipynb_checkpoints
# !find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
# !find . -type d -name ".ipynb_checkpoints" -exec rm -r {} + 2>/dev/null

# # ✅ (Optional) Remove pip cache if needed
# !pip cache purge


# # ✅ Install requirements from a file one folder up
get_ipython().system('pip install -r ../requirements.txt')


# In[ ]:





# In[ ]:





# In[2]:


import pickle
import pandas as pd
# Load the saved model
with open("regression_trainer_model.pkl", "rb") as f:
    elastic_trainer = pickle.load(f)

# Print all attributes of the loaded object
attributes = {attr: getattr(elastic_trainer, attr) for attr in dir(elastic_trainer) if not attr.startswith("__")}
for key, value in attributes.items():
    print(f"{key}: {type(value)}")  # Print attribute names and types


# In[15]:


import pickle
import pandas as pd
# Load the saved model
with open("classifier_trainer_model.pkl", "rb") as f:
    classifier_trainer = pickle.load(f)

# Print all attributes of the loaded object
attributes = {attr: getattr(classifier_trainer, attr) for attr in dir(classifier_trainer) if not attr.startswith("__")}
for key, value in attributes.items():
    print(f"{key}: {type(value)}")  # Print attribute names and types

df_classifier_preds = classifier_trainer.classifier_predictions_df.copy()


# In[ ]:





# In[16]:


# Print the column names and preview the data to confirm structure
print("🔹 x_test_with_meta Columns:", elastic_trainer.x_test_with_meta.columns.tolist())
print("🔹 y_test (Actual Values) Sample:", elastic_trainer.y_test.tail(5))
print("🔹 Predictions Sample:", elastic_trainer.predictions[-5:])


# In[17]:


# Merge x_test_with_meta (Date, Time, OHLC) with y_test (Actual Next_High) and Predictions
df_test_results = elastic_trainer.x_test_with_meta[["Date", "Time", "Open", "High", "Low", "Close"]].copy()
df_test_results["Next_High"] = elastic_trainer.y_test.values  # Actual next high
df_test_results["Predicted_High"] = elastic_trainer.predictions  # Predicted next high

# Display the last 10 rows to inspect
import pandas as pd
df_test_results


# In[18]:


import matplotlib.pyplot as plt

# Calculate delta
df_test_results["Delta"] = df_test_results["Predicted_High"] - df_test_results["Close"]

bins = [-float("inf"), 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, float("inf")]
labels = ["<0", "0–0.5", "0.5–1", "1–1.5", "1.5–2", "2–2.5", "2.5–3", "3–3.5", "3.5–4", "4+"]

# Categorize
df_test_results["Delta_Bucket"] = pd.cut(df_test_results["Delta"], bins=bins, labels=labels, right=False)

# Count and sort
bucket_counts = df_test_results["Delta_Bucket"].value_counts().sort_index()

# Plot
plt.figure(figsize=(10, 4))
bucket_counts.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title("Delta Buckets (Predicted - Close)")
plt.xlabel("Delta Range (points)")
plt.ylabel("Number of Bars")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Optional: preview
df_test_results[["Date", "Time", "Close", "Predicted_High", "Delta", "Delta_Bucket"]].tail()


# In[19]:


# Step 1: Create a copy of the test set
df_test_results = elastic_trainer.x_test_with_meta.copy()
df_test_results["Predicted"] = elastic_trainer.predictions
df_test_results["Actual"] = elastic_trainer.y_test.values

# Step 2: Compute difference between prediction and close
df_test_results["Predicted_Diff"] = df_test_results["Predicted"] - df_test_results["Close"]

# Step 3: Filter only predictions >= 1.0
high_conf_preds = df_test_results[df_test_results["Predicted_Diff"] >= 1.0]

# Step 4: Determine how many of those actually reached >= Close + 1.0
high_conf_preds["Is_Good"] = high_conf_preds["Actual"] >= (high_conf_preds["Close"] + 1.0)

# Step 5: Count results
total_candidates = len(high_conf_preds)
num_good = high_conf_preds["Is_Good"].sum()
percent_good = 100 * num_good / total_candidates if total_candidates > 0 else 0

print(f"📈 Total predictions >= 1.0 points: {total_candidates}")
print(f"✅ Number of 'good' bars (Actual >= Close + 1.0): {num_good}")
print(f"🎯 Percentage good: {percent_good:.2f}%")


# In[20]:


# Step 1: Create prediction - close column
df_test_results['prediction_minus_close'] = df_test_results['Predicted'] - df_test_results['Close']

# Step 2: Filter for 1+ point trades
df_above_1 = df_test_results[df_test_results['prediction_minus_close'] >= 1.0].copy()

# Step 3: Convert "Time" to hour
df_above_1['Hour'] = pd.to_datetime(df_above_1['Time'], format='%H:%M').dt.hour

# Step 4: Group into 2-hour bins
bins = list(range(0, 25, 2))  # 0, 2, 4, ..., 22, 24
labels = [f"{h:02d}-{h+2:02d}" for h in bins[:-1]]
df_above_1['HourRange'] = pd.cut(df_above_1['Hour'], bins=bins, labels=labels, right=False)

# Step 5: Count occurrences
time_distribution = df_above_1['HourRange'].value_counts().sort_index()

# Step 6: Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
time_distribution.plot(kind='bar', color='skyblue')
plt.title("📊 Number of 1+ Point Trades by 2-Hour Windows")
plt.xlabel("Time Window")
plt.ylabel("Number of Trades")
plt.grid(True, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[21]:


import matplotlib.pyplot as plt
import numpy as np

# Step 1: Count trades per 2-hour window
df_above_1['Time Window'] = pd.cut(
    df_above_1['Hour'],
    bins=list(range(0, 26, 2)),  # 0-2, 2-4, ..., 22-24
    labels=[f"{str(i).zfill(2)}-{str(i+2).zfill(2)}" for i in range(0, 24, 2)],
    right=False
)

# Step 2: Count occurrences per time window
trade_counts = df_above_1['Time Window'].value_counts().sort_index()

# Step 3: Compute number of unique days
num_days = df_above_1['Date'].nunique()

# Step 4: Compute average trades per day
trade_avg_per_day = trade_counts / num_days

# Step 5: Plot both bars
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.4  # Adjust spacing between bars

# Plot total count
ax.bar(np.arange(len(trade_counts)) - bar_width/2, trade_counts, width=bar_width, alpha=0.6, label="Total Count", color="skyblue")

# Plot average per day
ax.bar(np.arange(len(trade_avg_per_day)) + bar_width/2, trade_avg_per_day, width=bar_width, alpha=0.9, label="Avg per Day", color="orange")

# Formatting
ax.set_xticks(np.arange(len(trade_counts)))
ax.set_xticklabels(trade_counts.index, rotation=45)
ax.set_ylabel("Number of Trades")
ax.set_xlabel("Time Window")
ax.set_title("📊 Number of 1+ Point Trades by 2-Hour Windows (Total vs Avg per Day)")
ax.legend()

plt.show()


# In[ ]:





# ### 📘 **Backtest Strategy Overview**
# 
# This strategy evaluates the effectiveness of our **ElasticNet model predictions** using the **Backtrader** framework.
# 
# ---
# 
# #### **🔹 Entry Logic**
# - **Only long trades**
# - Enter when:
#   - **Time** is within a configurable window (default: `17:00–23:00`)
#   - **Predicted High** is between a **minimum** and **maximum** distance above the **current close**
#   - **No active trade** is open (non-overlapping logic)
# 
# ---
# 
# #### **🔹 Execution Rules**
# - Entry is made at the **open of the next bar**
# - **Profit Target** and **Stop Loss** are fixed and defined in **ticks**
#   - _Example_: 10 ticks = 2.5 points
# - Optional: Force **exit at session end (23:00)**
# 
# ---
# 
# #### **🔹 Configurable Inputs**
# - Entry time window
# - Minimum/maximum delta (predicted - close)
# - Profit target (in ticks)
# - Stop loss (in ticks)
# - Tick size (default: 0.25)
# - Tick value (default: $1.25)
# - Force exit at session close
# 
# ---
# 
# This provides a clean, repeatable structure to test predictive strategies.
# 

# In[22]:


get_ipython().system('pip install backtrader')
get_ipython().system('pip install plotly')


# 

# In[ ]:





# In[24]:


import backtrader as bt
import pandas as pd
from datetime import time, datetime

class ElasticNetStrategy(bt.Strategy):
    params = dict(
        min_dist=3,
        max_dist=20.0,
        target_ticks=10,
        stop_ticks=10,
        slippage=0.0,
        force_exit=True,
        session_start='10:00',
        session_end='23:00',
        tick_value=1.25,
        contract_size=1,
        min_classifier_signals=0

    )

    def __init__(self):
        self.order = None
        self.entry_price = None
        self.open_trade_time = None
        self.trades = []

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        current_time = dt.strftime('%H:%M')

        # Exit on session end
        if self.position and self.p.force_exit and current_time >= self.p.session_end:
            print(f"[{dt}] ⏹️ Session end. Closing position.")
            self.close()
            self.order = None
            return

        if current_time < self.p.session_start or current_time >= self.p.session_end:
            return

        if self.position or self.order:
            return

        close = self.data.close[0]
        predicted = self.data.predicted_high[0]
        delta = predicted - close

        print(f"[{dt}] Close: {close:.2f}, Predicted: {predicted:.2f}, Delta: {delta:.2f}")
        # ✅ Validate classifier signal requirement
        if self.p.min_classifier_signals > 0:
            try:
                rf_val = self.data.RandomForest[0]
                lt_val = self.data.LightGBM[0]
                xg_val = self.data.XGBoost[0]

                if any(map(pd.isna, [rf_val, lt_val, xg_val])):
                    print(f"[{dt}] 🕳️ Skipping bar — classifier signal is NaN")
                    return

                rf = int(rf_val)
                lt = int(lt_val)
                xg = int(xg_val)

            except Exception as e:
                raise ValueError(f"❌ Error accessing classifier columns: {e}")


            green_count = rf + lt + xg

            if self.p.min_classifier_signals > 3:
                raise ValueError("❌ min_classifier_signals cannot be greater than 3.")

            if green_count < self.p.min_classifier_signals:
                print(f"[{dt}] 🚫 Not enough green signals ({green_count}) for entry.")
                return

        tick_size = 0.25
        if self.p.min_dist <= delta <= self.p.max_dist:
            entry_price = self.data.open[1] + self.p.slippage
            stop_price = entry_price - (self.p.stop_ticks * tick_size)
            target_price = entry_price + (self.p.target_ticks * tick_size)

            print(f"💥 Entry signal | Entry: {entry_price:.2f}, TP: {target_price:.2f}, SL: {stop_price:.2f}")

            self.order = self.buy_bracket(
                price=entry_price,
                size=1,
                stopprice=stop_price,
                limitprice=target_price
            )
            self.entry_price = entry_price
            self.open_trade_time = dt
    def next_open(self):
        dt = self.datas[0].datetime.datetime(0)
        current_time = dt.strftime('%H:%M')

        if self.position or self.order:
            return

        if current_time < self.p.session_start or current_time >= self.p.session_end:
            return

        close = self.data.close[0]
        predicted = self.data.predicted_high[0]
        delta = predicted - close

        if self.p.min_dist <= delta <= self.p.max_dist:
            entry_price = self.data.open[0] + self.p.slippage
            stop_price = entry_price - (self.p.stop_ticks * 0.25)
            target_price = entry_price + (self.p.target_ticks * 0.25)

            self.order = self.buy_bracket(
                price=entry_price,
                size=1,
                stopprice=stop_price,
                limitprice=target_price
            )
            self.entry_price = entry_price
            self.open_trade_time = dt

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"✅ BUY EXECUTED @ {order.executed.price:.2f}")
                self.entry_price = order.executed.price
                self.open_trade_time = self.data.datetime.datetime(0)

            elif order.issell():
                self.log(f"🏁 SELL EXECUTED @ {order.executed.price:.2f}")
                tick_size = 0.25  # or make this a param later
                ticks_moved = (order.executed.price - self.entry_price) / tick_size
                pnl = ticks_moved * self.p.contract_size * self.p.tick_value

                self.trades.append({
                    "entry_time": self.open_trade_time,
                    "exit_time": self.data.datetime.datetime(0),
                    "entry_price": self.entry_price,
                    "exit_price": order.executed.price,
                    "pnl": pnl
                })

            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"❌ Order Failed: {order.Status[order.status]}")
            self.order = None

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f"[{dt}] {txt}")


# 

# In[25]:


import backtrader as bt

class PandasData(bt.feeds.PandasData):
    lines = ('predicted_high',)  # Custom line

    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1),  # Required, even if unused
        ('predicted_high', -1),  # -1 means we provide it via 'lines' directly
    )

# 2. Initialize Cerebro
cerebro = bt.Cerebro()

# 3. Add our strategy
cerebro.addstrategy(ElasticNetStrategy)

df_bt = df_test_results.copy()
df_bt['datetime'] = pd.to_datetime(df_bt['Date'] + ' ' + df_bt['Time'])
df_bt.set_index('datetime', inplace=True)
# Ensure both are using datetime index
df_classifier_preds.index = pd.to_datetime(df_classifier_preds.index)
df_bt.index = pd.to_datetime(df_bt.index)

# ✅ Merge classifier signals into df_bt
df_bt = df_bt.merge(df_classifier_preds, how="left", left_index=True, right_index=True)

# Must match column name with custom line
df_bt['predicted_high'] = df_bt['Predicted']


# 5. Define a custom feed class that includes classifier columns
class CustomClassifierData(bt.feeds.PandasData):
    lines = ('predicted_high', 'RandomForest', 'LightGBM', 'XGBoost')
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1),
        ('predicted_high', -1),
        ('RandomForest', -1),
        ('LightGBM', -1),
        ('XGBoost', -1),
    )

# ✅ Get the required param from the strategy class

# Get the earliest timestamp that has classifier predictions
if ElasticNetStrategy.params.min_classifier_signals > 0:
    classifier_start = df_classifier_preds.index.min()
    print(f"⏳ Using classifier data from: {classifier_start}")

    # Trim df_bt to only rows from that timestamp onwards
    df_bt = df_bt[df_bt.index >= classifier_start]



# 6. Feed the data using the custom class
data_feed = CustomClassifierData(dataname=df_bt)
cerebro.adddata(data_feed)
# 🧠 Set initial cash
cerebro.broker.setcash(10000.0)

# ✅ Use tick_value and contract size to scale trades properly
# Each contract represents $1.25 per tick = $5 per point (4 ticks)
cerebro.broker.setcommission(
    commission=0.0,  # or non-zero if needed
    mult=1 * 1.25 / 0.25  # = 5.0  (multiplier = contract size * tick_value / tick_size)
)

# 7. Run the backtest
results = cerebro.run()



# In[26]:


print(df_bt[["RandomForest", "LightGBM", "XGBoost"]].notna().sum())
print(df_bt[["RandomForest", "LightGBM", "XGBoost"]].tail(5))


# In[ ]:





# In[27]:


# Grab the first (and only) strategy instance
strat = results[0]


print(f"📦 Final Portfolio Value: {cerebro.broker.getvalue():.2f}")


# Check number of closed trades
print(f"✅ Total Closed Trades: {len(strat.trades)}")


# In[28]:


# Ensure Datetime column exists
df_test_results["Datetime"] = pd.to_datetime(df_test_results["Date"] + " " + df_test_results["Time"])
df_test_results.set_index("Datetime", inplace=True)

# Rebuild OHLC
ohlc_df = df_test_results[["Open", "High", "Low", "Close"]].rename(
    columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
)

# Convert trade list to DataFrame
trade_df = pd.DataFrame(results[0].trades)

# Filter only trades with valid entry and exit timestamps
valid_trades = trade_df[
    trade_df["entry_time"].isin(ohlc_df.index) & trade_df["exit_time"].isin(ohlc_df.index)
].copy()

print(f"✅ Valid trades for plotting: {len(valid_trades)} / {len(trade_df)}")

# Save for plotting
buy_df = valid_trades.set_index("entry_time")
sell_df = valid_trades.set_index("exit_time")


# ## 📈 Trade & Prediction Chart
# 
# This chart visualizes:
# 
# - **Candlesticks**: Bar-by-bar price movement (Open, High, Low, Close)
# - **Actual vs. Predicted Highs**: Overlayed lines with annotations
# - **Trade Markers**: 
#   - 🟢 **Buy** = green upward triangle
#   - 🔴 **Sell** = red downward triangle
# 
# This view helps evaluate the accuracy of predictions and the timing & outcome of each trade.
# 

# In[29]:


import plotly.graph_objects as go
import pandas as pd

# ✅ Rebuild plotting DataFrame
plot_df = df_test_results.copy()
plot_df["Timestamp"] = pd.to_datetime(plot_df["Date"] + " " + plot_df["Time"])
plot_df["Actual_High"] = plot_df["High"].shift(-1)
plot_df = plot_df.dropna(subset=["Actual_High"])
plot_df.set_index("Timestamp", inplace=True)

# ✅ Merge classifier signals again just in case
plot_df = plot_df.merge(df_classifier_preds, how="left", left_index=True, right_index=True)

# ✅ Add classifier marker Y offsets
buffer = (plot_df["High"].max() - plot_df["Low"].min()) * 0.01
plot_df["rf_y"] = plot_df["Low"] - buffer
plot_df["lt_y"] = plot_df["Low"] - buffer * 2
plot_df["xg_y"] = plot_df["Low"] - buffer * 3

# 🔍 Confirm classifier columns present
print("🧩 Columns in plot_df:", plot_df.columns)

# ✅ Create base chart
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=plot_df.index,
    open=plot_df["Open"],
    high=plot_df["High"],
    low=plot_df["Low"],
    close=plot_df["Close"],
    name="Candlestick",
    increasing_line_color='lime',
    decreasing_line_color='red',
    line=dict(width=1),
    opacity=1.0
))

# ✅ Add predicted and actual highs
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
    line=dict(color="skyblue", dash='dash'),
    marker=dict(symbol='circle', size=6)
))

# ✅ Add trade markers with price text
visible_trades = 0
for _, trade in valid_trades.iterrows():
    entry = trade["entry_time"]
    exit_ = trade["exit_time"]
    eprice = trade["entry_price"]
    xprice = trade["exit_price"]

    if entry in plot_df.index and exit_ in plot_df.index:
        visible_trades += 1
        fig.add_trace(go.Scatter(
            x=[entry],
            y=[eprice],
            mode='markers+text',
            name='Buy',
            text=[f'Buy @ {eprice:.2f}'],
            textposition='bottom center',
            marker=dict(symbol='triangle-up', size=12, color='lime'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[exit_],
            y=[xprice],
            mode='markers+text',
            name='Sell',
            text=[f'Sell @ {xprice:.2f}'],
            textposition='top center',
            marker=dict(symbol='triangle-down', size=12, color='red'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[entry, exit_],
            y=[eprice, xprice],
            mode='lines',
            line=dict(color='yellow', width=2),
            showlegend=False
        ))

# ✅ Plot classifier markers with red/green color
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

print(f"🧮 Trades visible in Plotly chart: {visible_trades} / {len(valid_trades)}")

# ✅ Layout styling
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


# In[30]:


print("✔️ RandomForest:", plot_df["RandomForest"].dropna().unique())
print("✔️ LightGBM:", plot_df["LightGBM"].dropna().unique())
print("✔️ XGBoost:", plot_df["XGBoost"].dropna().unique())


# In[31]:


# 1. Get the strategy instance (usually one if not optimizing)
strat = results[0]

# 2. Get the trade list
trade_list = strat.trades

# 3. Display as DataFrame with $ formatting
import pandas as pd

df_trades = pd.DataFrame(trade_list)

# ✅ Format the PnL column with a dollar sign
df_trades["pnl"] = df_trades["pnl"].apply(lambda x: f"${x:.2f}")

df_trades


# In[32]:


trade_list


# In[ ]:





# In[ ]:





# In[33]:


import plotly.graph_objects as go
import pandas as pd


df_trades["pnl"] = df_trades["pnl"].replace('[\$,]', '', regex=True).astype(float)

# Compute equity
df_trades["equity"] = df_trades["pnl"].cumsum()

# Compute drawdown
df_trades["cum_max"] = df_trades["equity"].cummax()
df_trades["drawdown"] = df_trades["equity"] - df_trades["cum_max"]
max_dd = df_trades["drawdown"].min()
end_idx = df_trades["drawdown"].idxmin()
start_idx = df_trades["equity"][:end_idx].idxmax()

# Create plot
fig = go.Figure()

# Plot equity line
fig.add_trace(go.Scatter(
    x=df_trades.index,
    y=df_trades["equity"],
    mode="lines+markers",
    name="Equity Curve",
    line=dict(color="royalblue")
))

# Annotate max drawdown
fig.add_trace(go.Scatter(
    x=[start_idx, end_idx],
    y=[df_trades.loc[start_idx, "equity"], df_trades.loc[end_idx, "equity"]-25],
    mode="lines+text",
    line=dict(color="red", dash="dash", width=2),
    name="Max Drawdown",
    text=[None, f"⬇️ Max DD: ${abs(max_dd):.2f}"],
    textposition="top center"
))

# Layout
fig.update_layout(
    title="📉 Equity Curve with Max Drawdown",
    xaxis_title="Trade Index",
    yaxis_title="Cumulative PnL ($)",
    template="plotly_white",
    height=500,
    showlegend=True
)

fig.show()


# In[34]:


# ✅ Ensure datetime format
df_trades["entry_time"] = pd.to_datetime(df_trades["entry_time"])
df_trades["exit_time"] = pd.to_datetime(df_trades["exit_time"])

# ✅ Calculate difference in bars (5 minutes each)
df_trades["bars_held"] = (df_trades["exit_time"] - df_trades["entry_time"]) / pd.Timedelta(minutes=5)

# ✅ Count how many trades exited after exactly 1 bar
one_bar_trades = df_trades[df_trades["bars_held"] == 1]
print(f"📊 Trades exited after 1 bar: {len(one_bar_trades)} / {len(df_trades)}")

# Optional: view them
one_bar_trades.head()


# In[ ]:





# In[ ]:





# In[ ]:





# # Simulated Intrabar Exit Using 1-Minute Data
# This section adds a new strategy that:
# 
# Keeps using 5-min bars for entries
# 
# Uses 1-min bars (data1) to check if TP or SL were hit intrabar
# 
# Does not place bracket orders, but handles TP/SL exits manually
# 
# Keeps the original strategy and setup intact
# 
# 

# In[35]:


import pandas as pd

# Load the 1-minute file
df_1min = pd.read_csv("MES_1_MINUTE_JAN_13_JAN_21.txt", sep=",", parse_dates=[["Date", "Time"]])

# Set datetime index
df_1min.set_index("Date_Time", inplace=True)

# Preview structure
df_1min.head()


# In[36]:


# Show 5-min structure
print("🕔 5-Min DataFrame (df_test_results):")
print(df_test_results[["Date", "Time", "Open", "High", "Low", "Close"]].head(), "\n")

# Show 1-min structure
print("🕐 1-Min DataFrame (df_1min):")
print(df_1min.head())


# In[37]:


import pandas as pd

# Ensure datetime indexes
df_test_results.index = pd.to_datetime(df_test_results.index)
df_1min.index = pd.to_datetime(df_1min.index)
df_1min = df_1min.sort_index()

# Step 1: Filter 5-min bars that end no later than 23:00
valid_5min_bars = df_test_results[df_test_results.index.time <= pd.to_datetime("23:00").time()]

# Step 2: Generate expected 1-min timestamps for each valid 5-min bar
expected_1min_timestamps = []
for ts in valid_5min_bars.index:
    expected_1min_timestamps.extend(pd.date_range(start=ts, periods=5, freq="T"))

expected_1min_set = set(expected_1min_timestamps)
existing_1min_set = set(df_1min.index)

# Step 3: Fill missing 23:00 if 22:59 exists
missing = sorted(expected_1min_set - existing_1min_set)
filled = []

for ts in missing:
    if ts.strftime('%H:%M') == '23:00':
        ts_prev = ts - pd.Timedelta(minutes=1)
        if ts_prev in df_1min.index:
            df_1min.loc[ts] = df_1min.loc[ts_prev]
            filled.append(ts)

# Step 4: Re-check missing after fill
remaining_missing = sorted(set(expected_1min_timestamps) - set(df_1min.index))

# Step 5: Output
if not remaining_missing:
    print("✅ All required 1-min bars are present.")
else:
    print("❌ Still missing the following timestamps:")
    for m in remaining_missing:
        print(m)


# In[38]:


import pandas as pd

# Ensure datetime indexes and sorted
df_test_results.index = pd.to_datetime(df_test_results.index)
df_1min.index = pd.to_datetime(df_1min.index)
df_1min = df_1min.sort_index()

# Filter only 5-min bar timestamps from df_test_results
expected_5min_times = df_test_results.index

# Track missing 5-min timestamps that are before or at 23:00
missing_bars = []
for ts in expected_5min_times:
    if ts.time() <= pd.to_datetime("23:00").time():
        minute_range = pd.date_range(start=ts, periods=5, freq="T")
        if not all(minute in df_1min.index for minute in minute_range):
            missing_bars.append(ts)

# Build summary grouped by date
missing_summary = pd.DataFrame(missing_bars, columns=["Missing 5-Min Bar"])
missing_summary["Date"] = missing_summary["Missing 5-Min Bar"].dt.date
missing_summary["Time"] = missing_summary["Missing 5-Min Bar"].dt.strftime('%H:%M')

grouped = missing_summary.groupby("Date")["Time"].apply(lambda x: ", ".join(x)).reset_index()
grouped.columns = ["Date", "Missing Times (<=23:00)"]

grouped.head(20)


# In[ ]:





# In[ ]:





# In[40]:


class ElasticNetIntrabarStrategy(bt.Strategy):
    params = dict(
        min_dist=3.0,
        max_dist=20.0,
        target_ticks=10,
        stop_ticks=10,
        slippage=0.0,
        force_exit=True,
        session_start='10:00',
        session_end='23:00',
        tick_value=1.25,
        contract_size=1,
        min_classifier_signals=3,
        tick_size=0.25,
    )

    def __init__(self):
        self.order = None
        self.entry_price = None
        self.open_trade_time = None
        self.trades = []
        self.last_entry_time = None


    def next(self):
        dt_5min = self.datas[0].datetime.datetime(0)
        dt_1min = self.datas[1].datetime.datetime(0)

        print(f"🧠 Bar time: {dt_5min} | {dt_1min}")
        print(f"🧪 Checking bar: {dt_5min} → minute = {dt_5min.minute}")

        # ✅ Skip if already processed this 5-min bar
        if self.last_entry_time == dt_5min:
            return
        self.last_entry_time = dt_5min

        # ✅ Only allow entries at 5-minute intervals
        if dt_5min.minute % 5 != 0:
            print(f"🚫 Skipping non-5-min bar: {dt_5min}")
            return


        current_time = dt_5min.strftime('%H:%M')

        if self.position and self.p.force_exit and current_time >= self.p.session_end:
            self.close()
            self.order = None
            return

        if self.position or self.order:
            return

        main = self.datas[0]
        intrabar = self.datas[1]

        close = main.close[0]
        predicted = main.predicted_high[0]
        delta = predicted - close

        print(f"🔬 At {dt_5min} — Close: {close:.2f}, Predicted: {predicted:.2f}, Delta: {delta:.2f}")

        if self.p.min_classifier_signals > 0:
            try:
                green_count = sum(int(self.datas[0].__getattr__(clf)[0]) for clf in ['RandomForest', 'LightGBM', 'XGBoost'])
            except Exception:
                return
            if green_count < self.p.min_classifier_signals:
                return

        if self.p.min_dist <= delta <= self.p.max_dist:
            print(f"🔍 Trade candidate found at {dt_5min}, delta: {delta:.2f}")
            entry_price = main.open[0] + self.p.slippage
            tp = entry_price + self.p.target_ticks * self.p.tick_size
            sl = entry_price - self.p.stop_ticks * self.p.tick_size
            entry_time = intrabar.datetime.datetime(0)

            for i in range(1, 6):
                if len(intrabar) <= i:
                    break
                hi = intrabar.high[i]
                lo = intrabar.low[i]
                print(f"⏱ Scanning 1-min at {intrabar.datetime.datetime(i)} → hi: {hi}, lo: {lo}")

                exit_price = None
                if lo <= sl:
                    exit_price = sl
                elif hi >= tp:
                    exit_price = tp

                if exit_price is not None:
                    ticks = (exit_price - entry_price) / self.p.tick_size
                    pnl = ticks * self.p.contract_size * self.p.tick_value
                    self.trades.append({
                        "entry_time": entry_time,
                        "exit_time": intrabar.datetime.datetime(i),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl
                    })
                    print(f"✅ Trade recorded — ENTRY: {entry_time}, EXIT: {intrabar.datetime.datetime(i)}, PnL: {pnl:.2f}")
                    return






# In[41]:


class IntrabarClassifierFeed(bt.feeds.PandasData):
    lines = ('predicted_high', 'RandomForest', 'LightGBM', 'XGBoost')
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1),
        ('predicted_high', -1),
        ('RandomForest', -1),
        ('LightGBM', -1),
        ('XGBoost', -1),
    )


# In[ ]:





# In[42]:


import backtrader as bt

# 🔧 Create feed for 5-min data (regression only)
class RegressionFeed(bt.feeds.PandasData):
    lines = ('predicted_high',)
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1),
        ('predicted_high', -1),
    )

# 🔁 Create new Cerebro instance to avoid overwriting old one
cerebro_intrabar = bt.Cerebro()
cerebro_intrabar.broker.setcash(10000.0)
cerebro_intrabar.broker.setcommission(commission=0.0, mult=5.0)  # 1 contract = $5 per point
# Convert and align indices
df_classifier_preds.index = pd.to_datetime(df_classifier_preds.index)
df_test_results.index = pd.to_datetime(df_test_results.index)

# 🔍 DEBUG: Print current classifier-related columns before any drop or merge
print("📋 BEFORE MERGE — df_test_results columns:")
print([col for col in df_test_results.columns if "RandomForest" in col or "LightGBM" in col or "XGBoost" in col])

print("\n📋 df_classifier_preds columns:")
print(df_classifier_preds.columns.tolist())

# 🔍 Check sample classifier values from classifier_preds (to confirm they are valid)
print("\n🔎 Sample values from df_classifier_preds:")
print(df_classifier_preds.dropna().head(3))

# 🔍 Print shape before merging to ensure data volume alignment
print(f"\n🧮 Shape before merge — df_test_results: {df_test_results.shape}, df_classifier_preds: {df_classifier_preds.shape}")


# 🧹 Drop all existing classifier-related columns to avoid duplicate suffix conflicts
classifier_cols = [col for col in df_test_results.columns if any(clf in col for clf in ["RandomForest", "LightGBM", "XGBoost"])]
df_test_results.drop(columns=classifier_cols, inplace=True, errors="ignore")

# ✅ Now safely merge classifier predictions
df_test_results = df_test_results.merge(
    df_classifier_preds, how="left", left_index=True, right_index=True
)

# 👁️ Optional: print to confirm
print("✅ After clean merge, classifier columns:")
print([col for col in df_test_results.columns if col in ["RandomForest", "LightGBM", "XGBoost"]])

df_test_results["predicted_high"] = df_test_results["Predicted"]
# ✅ Trim df_test_results if using classifier signals
if ElasticNetIntrabarStrategy.params.min_classifier_signals > 0:
    classifier_start = df_classifier_preds.index.min()
    print(f"⏳ Using classifier data from: {classifier_start}")
    df_test_results = df_test_results[df_test_results.index >= classifier_start]



# 📈 Add strategy
cerebro_intrabar.addstrategy(ElasticNetIntrabarStrategy)

# 📦 Feed both data streams
data_5min = IntrabarClassifierFeed(dataname=df_test_results)
data_1min = bt.feeds.PandasData(dataname=df_1min)

cerebro_intrabar.adddata(data_5min)  # Main feed
cerebro_intrabar.adddata(data_1min)  # Intrabar OHLC

# ▶️ Run strategy
results_intrabar = cerebro_intrabar.run()


# In[43]:


# 1. Get the strategy instance (usually one if not optimizing)
stra_intrabar = results_intrabar[0]

# 2. Get the trade list
trade_list_intrabar = stra_intrabar.trades


df_trades_intrabar = pd.DataFrame(trade_list_intrabar)


# ✅ Format the PnL column with a dollar sign
df_trades_intrabar["pnl"] = df_trades_intrabar["pnl"].apply(lambda x: f"${x:.2f}")

df_trades_intrabar


# In[44]:


df_test_results["predicted_high"] = df_test_results["Predicted"]

print("✅ 5-min data columns:", df_test_results.columns)
print("✅ Sample classifier values:")
print(df_test_results[["RandomForest", "LightGBM", "XGBoost"]].dropna().head(5))


# In[ ]:





# In[ ]:





# In[ ]:





# # Saving percition data for Tradestation analysis

# In[45]:


import re
import json
import pandas as pd

# Read JS content
with open("../frontend/src/components/initialData.js", "r") as f:
    js_content = f.read()

# Extract the three arrays
def extract_json_array(label):
    pattern = rf"export let {label} = (\[.*?\]);"
    match = re.search(pattern, js_content, re.DOTALL)
    return json.loads(match.group(1)) if match else []

# Extract and load
pred_data = pd.DataFrame(extract_json_array("PredActualData"))
clf_data = pd.DataFrame(extract_json_array("classifierData"))

# Merge on datetime
merged = pd.merge(pred_data, clf_data, on="date", how="outer")

# Save to CSV for TradeStation
merged.to_csv("ts_prediction_data.csv", index=False)


# In[46]:


merged.head()


# # Create Notbook as .py File

# In[47]:


get_ipython().system('jupyter nbconvert --to script intra_analyzer.ipynb')


# In[ ]:




