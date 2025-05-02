from datetime import datetime
import os
import json
import pyperclip

class ResearchAnalyzer:
    def __init__(self, config, fetchers):
        self.config = config
        self.fetchers = fetchers
        self.outputs = {
            "metadata": {},
            "general": {},
            "symbols": {}
        }

    def build_metadata(self):
        """
        Create metadata block with generation time and user preferences.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.outputs["metadata"] = {
            "generation_time": now,
            "requested_markets": self.config["markets"],
            "requested_symbols": self.config["symbols"]
        }

    def analyze_general(self):
        """
        Fetch news, smart money, economic events and generate general market insights.
        """
        news_headlines = self.fetchers.fetch_news_headlines(
            self.config["markets"], 
            self.config["symbols"]
        )

        smart_money_trend, top_sectors = self.fetchers.fetch_smart_money_flow()
        economic_events = self.fetchers.fetch_economic_events()

        # Summarize bias based on simple logic
        if "Fed" in " ".join([e["event"] for e in economic_events]):
            general_bias = "Cautious"
        elif smart_money_trend == "Accumulation":
            general_bias = "Bullish"
        elif smart_money_trend == "Distribution":
            general_bias = "Bearish"
        else:
            general_bias = "Neutral"

        self.outputs["general"] = {
            "general_bias": general_bias,
            "smart_money_flow": {
                "trend": smart_money_trend,
                "top_sectors": top_sectors
            },
            "economic_calendar": economic_events,
            "headline_sample": news_headlines[:5]  # First 5 headlines
        }

    def analyze_symbols(self):
        """
        Analyze requested symbols one by one.
        """
        symbols = self.config["symbols"]
        for symbol in symbols:
            # Simplified: check if any news contains the symbol
            found = False
            symbol_summary = {
                "sentiment": "Unknown",
                "unusual_volume": False,
                "smart_money_flow": "Unknown",
                "recommended_action": "No strong recommendation"
            }

            # Simulate symbol search in headlines
            sample_headlines = self.outputs["general"].get("headline_sample", [])
            for headline in sample_headlines:
                if symbol in headline:
                    found = True
                    symbol_summary["sentiment"] = "Positive"
                    symbol_summary["recommended_action"] = "Watch for Long Opportunity"
                    break

            if not found:
                symbol_summary["status"] = "No fresh news found today."

            self.outputs["symbols"][symbol] = symbol_summary

    def save_report(self):
        """
        Saves the current analysis output (self.outputs) as a JSON file.
        Now includes summarized_event_text from config, if available.
        """
        # 📂 Ensure output folder exists
        os.makedirs("research_outputs", exist_ok=True)
    
        # 🗓️ File name with today's date
        today_str = datetime.now().strftime("%Y-%m-%d")
        file_path = f"research_outputs/research_{today_str}.json"
    
        # ✅ Add summarized LLM economic event text to outputs if available
        if "general" in self.outputs and "summarized_events" in self.config:
            self.outputs["general"]["summarized_event_text"] = self.config["summarized_events"]
    
        # 💾 Save the output to JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.outputs, f, ensure_ascii=False, indent=2)
    
        print(f"✅ Research report saved to {file_path}.")

    def print_summary(self):
        """
        Print live short summary and optionally copy to clipboard.
        """
        bias = self.outputs["general"].get("general_bias", "Unknown")
        sectors = self.outputs["general"].get("smart_money_flow", {}).get("top_sectors", [])
        upcoming_events = [e["event"] for e in self.outputs["general"].get("economic_calendar", [])]

        print("\n🔔 Daily Summary:")
        print(f"🧠 Market Bias: {bias}")
        print(f"🏛️ Focus Sectors: {', '.join(sectors) if sectors else 'None'}")
        print(f"📅 Key Events: {', '.join(upcoming_events) if upcoming_events else 'None'}\n")

        if self.config["copy_to_clipboard"]:
            summary_text = f"Bias: {bias} | Sectors: {', '.join(sectors)} | Events: {', '.join(upcoming_events)}"
            pyperclip.copy(summary_text)
            print("📋 Summary copied to clipboard!")

    def run_daily_analysis(self):
        """
        Complete run: metadata → analysis → save → display
        """
        self.build_metadata()
        self.analyze_general()
        self.analyze_symbols()
        self.save_report()
        if self.config["print_live_summary"]:
            self.print_summary()

    def analyze_symbols(self):
        """
        Analyze each symbol separately: fetch news, fetch company profile, and prepare outputs.
        """
        self.outputs["symbols"] = {}
    
        for symbol in self.config.get("symbols", []):
            print(f"\n🔎 Analyzing Symbol: {symbol}")
    
            # Fetch top headlines
            headlines = self.fetchers.fetch_news_headlines([symbol])
    
            # Fetch company profile using FMP
            profile_data = self.fetchers.fetch_company_profile(symbol)
    
            # Optional: If you want to fetch sentiment separately later (not mandatory yet)
            # sentiment_data = self.fetchers.fetch_sentiment(symbol)
    
            # Build output for this symbol
            self.outputs["symbols"][symbol] = {
                "headline_sample": headlines[:5],
                "company_profile": profile_data,
                # "sentiment": sentiment_data,  # If you add it later
            }
    
            # Print a brief summary
            print(f"✅ {symbol}: Found {len(headlines)} headlines.")
            if profile_data:
                print(f"✅ {symbol}: Sector: {profile_data.get('sector', 'N/A')}, Industry: {profile_data.get('industry', 'N/A')}")

    def display_metrics_plotly(self, df):
        """
        Display metrics in a Plotly table.
        """
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

    def display_strategy_and_metrics_side_by_side(self, df_metrics, strategy_params):
        """
        Display strategy performance metrics and parameters side-by-side using Plotly tables.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd

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
            "⚠️ Risk / Drawdown Metrics | Max Consecutive Losses": lambda v: f"${v:.2f}",
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