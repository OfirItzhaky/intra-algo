# Import the numpy patch to fix NaN issue
import fix_numpy

from flask import Flask, request, render_template_string, redirect, url_for,jsonify
from datetime import datetime
import base64
import plotly.graph_objects as go
import pandas_ta as ta

from research_fetchers import ResearchFetchers, summarize_with_cache, summarize_economic_events_with_cache
from research_analyzer import ResearchAnalyzer
from image_analyzer_ai import ImageAnalyzerAI
from news_aggregator import NewsAggregator
import os
import pandas as pd
from templates import HTML_TEMPLATE
import traceback

# === Runtime Constants ===
today = datetime.today().strftime("%Y-%m-%d")
DEFAULT_MARKETS = ["US"]
DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "NVDA"]
FOCUS_SECTORS = ["Technology", "Healthcare", "Energy"]
ONLY_MAJOR_EVENTS = False
PRINT_LIVE_SUMMARY = True
COPY_TO_CLIPBOARD = False
SAVE_DIRECTORY = "research_outputs"

from config import CONFIG, SUMMARY_CACHE, EVENT_CACHE

app = Flask(__name__)
app.secret_key = 'snapshot-session-key'
# Enable debugging and detailed error display
app.config['DEBUG'] = True
app.config['PROPAGATE_EXCEPTIONS'] = True

# === Instantiate components ===
fetchers = ResearchFetchers(config=CONFIG)
aggregator = NewsAggregator(config=CONFIG)
analyzer = ResearchAnalyzer(config=CONFIG, fetchers=fetchers, aggregator=aggregator)

UPLOAD_FOLDER = os.path.abspath("uploaded_csvs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
symbol_data = {}  # global store
image_ai = ImageAnalyzerAI(
    model_provider=CONFIG["model_provider"],
    model_name=CONFIG["model_name"],
    api_key=CONFIG["openai_api_key"] if CONFIG["model_provider"] == "openai" else CONFIG["gemini_api_key"]
)

# === Prepare context ===
def prepare_daily_context():
    news_aggregator = NewsAggregator(config=CONFIG)
    merged_headlines = news_aggregator.aggregate_news()
    summarized_news = summarize_with_cache(fetchers, merged_headlines, force_refresh=False)
    summarized_events = summarize_economic_events_with_cache(fetchers, force_refresh=False)

    CONFIG["summarized_news"] = summarized_news
    CONFIG["summarized_events"] = summarized_events

# === Shared state ===
daily_results = []
image_results = []

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    print("===== UPLOAD_CSV ROUTE CALLED =====")
    print(f"Request method: {request.method}")
    print(f"Request files: {len(request.files)}")
    
    files = request.files.getlist("csv_files")
    
    # Instead of creating a new file_info list, retrieve the existing one or create empty
    file_info = app.config.get('FILE_INFO', []) 
    print(f"Existing file info entries: {len(file_info)}")
    
    for file in files:
        filename = file.filename
        try:
            print(f"Processing file: {filename}")
            # Extract symbol from filename 
            symbol = filename
            
            # Read the file directly with pandas
            try:
                # Check file extension to determine how to read it
                if filename.lower().endswith(('.txt', '.csv')):
                    df = pd.read_csv(file)
                elif filename.lower().endswith('.xlsx') or filename.lower().endswith('.xls'):
                    df = pd.read_excel(file)
                else:
                    # Try to read as CSV by default
                    try:
                        df = pd.read_csv(file)
                    except Exception:
                        raise Exception("Unsupported file format. Please upload CSV, TXT, or Excel files.")
                
                print(f"Successfully read file. Shape: {df.shape}")
                
                # Validate data - check minimum requirements
                if df.shape[0] < 20:
                    raise Exception(f"Insufficient data points (minimum 20 required, found {df.shape[0]})")
                
            except Exception as e:
                print(f"Error reading file: {str(e)}")
                raise Exception(f"Failed to parse file: {str(e)}")
            
            if len(df.columns) > 0:
                df.columns = [col.strip().capitalize() for col in df.columns]
                print(f"Columns: {list(df.columns)}")
                
                # Try to identify date column
                date_columns = [col for col in df.columns if any(date_term in col.lower() 
                                                            for date_term in ['date', 'time', 'datetime'])]
                
                if date_columns:
                    date_col = date_columns[0]
                    print(f"Found date column: {date_col}")
                    # Convert to datetime if it's not already
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    # Sort by date to ensure we get the correct last date
                    df = df.sort_values(by=date_col)
                elif isinstance(df.index, pd.DatetimeIndex):
                    print("Using DatetimeIndex")
                    # If the index is a datetime index
                    date_col = 'index'
                else:
                    # Try to convert the first column to datetime as a fallback
                    try:
                        first_col = df.columns[0]
                        print(f"Trying to convert first column '{first_col}' to datetime")
                        df[first_col] = pd.to_datetime(df[first_col], errors='coerce')
                        if not df[first_col].isna().all():  # At least some values converted successfully
                            date_col = first_col
                            print(f"Successfully used first column as date")
                        else:
                            date_col = None
                            print("First column cannot be parsed as dates")
                    except Exception as date_err:
                        print(f"Error with first column: {str(date_err)}")
                        date_col = None
                
                # Determine time interval
                interval_status = "valid"
                interval_message = ""
                
                if date_col:
                    if date_col == 'index':
                        dates = df.index
                    else:
                        dates = df[date_col].dropna()  # Remove NaT values
                    
                    print(f"Found {len(dates)} valid dates")
                    num_days = (max(dates) - min(dates)).days if len(dates) > 1 else 0
                    
                    if len(dates) > 1:
                        # Calculate the most common difference between consecutive dates
                        try:
                            # Sort dates to ensure proper differences
                            if isinstance(dates, pd.Series):
                                sorted_dates = dates.sort_values()
                            else:
                                sorted_dates = sorted(dates)
                                
                            # Calculate differences
                            date_diffs = []
                            for i in range(len(sorted_dates)-1):
                                try:
                                    diff = (sorted_dates.iloc[i+1] if hasattr(sorted_dates, 'iloc') else sorted_dates[i+1]) - \
                                           (sorted_dates.iloc[i] if hasattr(sorted_dates, 'iloc') else sorted_dates[i])
                                    date_diffs.append(diff.total_seconds())
                                except Exception as diff_err:
                                    print(f"Error calculating difference: {str(diff_err)}")
                            
                            if date_diffs:
                                # Most common difference in seconds
                                most_common_diff = pd.Series(date_diffs).mode()[0]
                                print(f"Most common diff: {most_common_diff} seconds")
                                
                                # Determine interval based on seconds
                                seconds_in_day = 86400
                                if abs(most_common_diff - seconds_in_day) < 60:  # Allow 1 minute tolerance
                                    interval = "Daily"
                                    years = num_days / 365
                                    if years >= 1:
                                        interval_message = f"{len(dates)} bars spanning {years:.1f} years"
                                    else:
                                        interval_message = f"{len(dates)} bars spanning {num_days} days"
                                elif abs(most_common_diff - seconds_in_day * 7) < 3600:  # Allow 1 hour tolerance
                                    interval = "Weekly"
                                    years = num_days / 365
                                    if years >= 1:
                                        interval_message = f"{len(dates)} bars spanning {years:.1f} years"
                                    else:
                                        weeks = num_days / 7
                                        interval_message = f"{len(dates)} bars spanning {int(weeks)} weeks"
                                elif (abs(most_common_diff - seconds_in_day * 30) < 3600 * 24 or 
                                    abs(most_common_diff - seconds_in_day * 31) < 3600 * 24):
                                    interval = "Monthly"
                                    years = num_days / 365
                                    if years >= 1:
                                        interval_message = f"{len(dates)} bars spanning {years:.1f} years"
                                    else:
                                        months = num_days / 30
                                        interval_message = f"{len(dates)} bars spanning {int(months)} months"
                                elif most_common_diff < seconds_in_day and most_common_diff >= 3600:
                                    hours = int(most_common_diff/3600)
                                    interval = f"{hours}h"
                                    # Flag intraday data that's not in standard intervals
                                    if hours not in [1, 2, 4, 6, 8, 12]:
                                        interval_status = "warning"
                                        interval_message = f"Non-standard intraday interval ({hours}h)"
                                    else:
                                        interval_message = f"{len(dates)} bars spanning {num_days} days"
                                elif most_common_diff < 3600 and most_common_diff >= 60:
                                    minutes = int(most_common_diff/60)
                                    interval = f"{minutes}m"
                                    # Flag intraday data that's not in standard intervals
                                    if minutes not in [1, 5, 15, 30]:
                                        interval_status = "warning"
                                        interval_message = f"Non-standard minute interval ({minutes}m)"
                                    else:
                                        interval_message = f"{len(dates)} bars spanning {num_days} days"
                                elif most_common_diff < 60:
                                    seconds = int(most_common_diff)
                                    interval = f"{seconds}s"
                                    interval_status = "warning"
                                    interval_message = "Second-based data may be too granular for analysis"
                                else:
                                    interval = f"Custom ({most_common_diff/seconds_in_day:.1f} days)"
                                    interval_status = "warning"
                                    interval_message = f"Non-standard interval ({most_common_diff/seconds_in_day:.1f} days)"
                                    
                                # Check for 0s interval which indicates potential data issues
                                if most_common_diff == 0:
                                    interval_status = "error"
                                    interval_message = "Invalid interval (0s) - data may have duplicate timestamps"
                            else:
                                print("No date diffs could be calculated")
                                interval = "Unknown"
                                interval_status = "error"
                                interval_message = "Could not determine data interval"
                        except Exception as diff_err:
                            print(f"Error determining interval: {str(diff_err)}")
                            interval = "Error determining interval"
                            interval_status = "error"
                            interval_message = f"Error: {str(diff_err)}"
                    else:
                        interval = "Unknown (not enough data points)"
                        interval_status = "error"
                        interval_message = "Not enough data points to determine interval"
                    
                    # Get last date
                    try:
                        if len(dates) > 0:
                            last_date = dates.iloc[-1] if hasattr(dates, 'iloc') else dates[-1]
                            last_date_str = last_date.strftime('%Y-%m-%d %H:%M')
                            print(f"Last date: {last_date_str}")
                            
                            # Check if data is recent enough (within last 30 days)
                            days_old = (datetime.now() - last_date).days
                            if days_old > 30:
                                interval_status = "warning" if interval_status == "valid" else interval_status
                                interval_message += f" Data is {days_old} days old."
                                
                        else:
                            last_date_str = "No valid dates found"
                            interval_status = "error"
                            interval_message = "No valid dates found in file"
                    except Exception as date_format_err:
                        print(f"Error getting last date: {str(date_format_err)}")
                        last_date_str = "Error getting last date"
                        interval_status = "error"
                        interval_message = f"Error getting last date: {str(date_format_err)}"
                else:
                    print("No date column identified")
                    interval = "Unknown (no date column found)"
                    last_date_str = "Unknown (no date column found)"
                    interval_status = "error"
                    interval_message = "No date column identified in data"
                
                # Store the DataFrame in symbol_data
                if symbol not in symbol_data:
                    symbol_data[symbol] = {}
                symbol_data[symbol][interval] = df
            else:
                interval = "Unknown (empty file)"
                last_date_str = "Unknown (empty file)"
                interval_status = "error"
                interval_message = "File contains no data columns"
            
            # Check if this file already exists in our list (by filename)
            existing_entry = next((item for item in file_info if item["symbol"] == filename), None)
            
            if existing_entry:
                # Update the existing entry
                existing_entry.update({
                    'interval': interval,
                    'last_bar_date': last_date_str,
                    'status': interval_status,
                    'message': interval_message
                })
                print(f"Updated existing entry for: {filename}")
            else:
                # Add new entry
                file_info.append({
                    'symbol': filename,
                    'interval': interval,
                    'last_bar_date': last_date_str,
                    'status': interval_status,
                    'message': interval_message
                })
                print(f"Added new entry for: {filename}")
            
            print(f"Successfully processed file: {filename}")
            
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            
            # Check if this file already exists in our list
            existing_error_entry = next((item for item in file_info if item["symbol"] == filename), None)
            
            if existing_error_entry:
                # Update the existing entry with the error
                existing_error_entry.update({
                    'interval': "Error",
                    'last_bar_date': f"Error: {str(e)}",
                    'status': "error",
                    'message': str(e)
                })
            else:
                # Add new error entry
                file_info.append({
                    'symbol': filename,
                    'interval': "Error",
                    'last_bar_date': f"Error: {str(e)}",
                    'status': "error",
                    'message': str(e)
                })
    
    # Save the updated file_info back to app config
    app.config['FILE_INFO'] = file_info
    print(f"Total files in memory after processing: {len(file_info)}")
    
    return redirect(url_for("index", no_reset="true"))

@app.route("/", methods=["GET"])
def index():
    print("===== INDEX ROUTE CALLED =====")
    
    # Perform a soft reset when the page is loaded
    # This ensures that refreshing the page gives a fresh start
    if request.args.get('no_reset') != 'true':
        print("Performing soft reset on page load")
        global daily_results, image_results, symbol_data
        
        # Clear in-memory data
        daily_results = []
        image_results = []
        symbol_data.clear()
        app.config['FILE_INFO'] = []
        
        # Don't clear API caches or delete files on page refresh
        # This is a "soft reset" - for full reset, user can click the Reset button
    
    # Use get() instead of pop() to preserve the file_info between requests
    file_info = app.config.get('FILE_INFO', [])
    upload_error = app.config.pop('UPLOAD_ERROR', None)
    
    print(f"File info: {file_info}")
    print(f"Upload error: {upload_error}")
    
    return render_template_string(
        HTML_TEMPLATE, 
        daily_outputs=daily_results, 
        image_outputs=image_results,
        file_info=file_info,
        upload_error=upload_error
    )

@app.route("/daily_analysis", methods=["POST"])
def daily_analysis():
    print("===== DAILY_ANALYSIS ROUTE CALLED =====")
    global daily_results
    
    try:
        prepare_daily_context()
        analyzer.run_daily_analysis()
        cost = fetchers.cost_usd

        market_bias = analyzer.outputs["general"].get("general_bias", "Unknown")
        sectors = ", ".join(analyzer.outputs["general"].get("smart_money_flow", {}).get("top_sectors", []))
        events = [e["event"] for e in analyzer.outputs["general"].get("economic_calendar", [])][:3]
        top_events = "\n".join(events)

        # Extract symbol info from self.outputs["symbols"]
        symbol_insights = []
        for symbol in DEFAULT_SYMBOLS:
            data = analyzer.outputs["symbols"].get(symbol, {})
            profile = data.get("company_profile", {})
            headlines = data.get("headline_sample", [])
            sector = profile.get("sector", "Unknown")
            industry = profile.get("industry", "Unknown")
            count = len(headlines)
            symbol_insights.append(f"{symbol} — {count} headlines — Sector: {sector}, Industry: {industry}")

        symbol_summary = "\n".join(symbol_insights)

        daily_results = [{
            "filename": "🧭 Daily Market Highlights",
            "text": f"**Bias:** {market_bias}\n**Focus Sectors:** {sectors}\n**Key Events:**\n{top_events}"
        }, {
            "filename": "📈 Symbol Insights",
            "text": symbol_summary
        }, {
            "filename": "📰 Full Daily Summary",
            "text": f"{CONFIG['summarized_news']}\n\n📅 Events:\n{CONFIG['summarized_events']}"
        }, {
            "filename": "💰 Cost Summary",
            "text": f"Total Estimated LLM Cost: ${cost:.6f}"
        }]

        return redirect(url_for("index", no_reset="true"))
        
    except Exception as e:
        error_details = traceback.format_exc()
        error_message = f"Error in daily analysis: {str(e)}\n\nStacktrace:\n{error_details}"
        print(error_message)  # Log to console/logs
        
        # Add error to daily_results
        daily_results = [{
            "filename": "❌ Error in Daily Analysis",
            "text": error_message
        }]
        
        # Return both HTML display and JSON error
        return render_template_string("""
        <html>
        <head>
            <title>Error in Daily Analysis</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body style="font-family: Arial; margin: 40px;">
            <div class="container">
                <div class="alert alert-danger">
                    <h3>Error in Daily Analysis</h3>
                    <p>{{ error }}</p>
                    <pre>{{ traceback }}</pre>
                </div>
                <a href="/" class="btn btn-primary">Return to Dashboard</a>
            </div>
        </body>
        </html>
        """, error=str(e), traceback=error_details)

@app.route("/momentum_analysis", methods=["POST"])
def momentum_analysis():
    print("===== MOMENTUM_ANALYSIS ROUTE CALLED =====")
    global daily_results

    try:
        have_uploaded_files = len(symbol_data) > 0

        if not have_uploaded_files:
            # Show a message if no files are uploaded
            return render_template_string("""
            <html>
            <head>
                <title>No Uploaded Files</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body style="font-family: Arial; margin: 40px;">
                <div class="container">
                    <div class="alert alert-warning">
                        <h3>No Uploaded Files</h3>
                        <p>Please upload data files to generate a momentum report.</p>
                    </div>
                    <a href="/" class="btn btn-primary">Return to Dashboard</a>
                </div>
            </body>
            </html>
            """)

        # First check if we have any uploaded files
        have_uploaded_files = len(symbol_data) > 0
        
        if have_uploaded_files:
            print("Found uploaded files, checking for symbol data")
            # Extract symbols from uploaded files
            symbols_from_files = []
            
            for filename in symbol_data.keys():
                base_name = os.path.splitext(filename)[0].lower()
                # Check for patterns like spy_daily.txt, qqq_weekly.txt
                if "_daily" in base_name:
                    symbol = base_name.replace("_daily", "").upper()
                    if symbol not in symbols_from_files:
                        symbols_from_files.append(symbol)
                elif "_weekly" in base_name:
                    symbol = base_name.replace("_weekly", "").upper()
                    if symbol not in symbols_from_files:
                        symbols_from_files.append(symbol)
            
            if symbols_from_files:
                print(f"Using symbols from uploaded files: {symbols_from_files}")
                # Create a subclass that will use our uploaded data instead of fetching
                class UploadedDataMomentumScorer(MarketMomentumScorer):
                    def fetch_data(self):
                        print("Using uploaded data instead of fetching from external sources")
                        # Process uploaded data
                        for symbol in self.symbols:
                            # Look for weekly data
                            weekly_df = None
                            daily_df = None
                            
                            # Check each uploaded file
                            for filename, data_dict in symbol_data.items():
                                file_lower = filename.lower()
                                if symbol.lower() in file_lower:
                                    for interval, df in data_dict.items():
                                        if interval == "Weekly" or "_weekly" in file_lower:
                                            weekly_df = df.copy()
                                            print(f"Using {filename} as weekly data for {symbol}")
                                        elif interval == "Daily" or "_daily" in file_lower:
                                            daily_df = df.copy()
                                            print(f"Using {filename} as daily data for {symbol}")
                            
                            # Store the data
                            if weekly_df is not None:
                                # Ensure column names are standardized
                                for col in ['Open', 'High', 'Low', 'Close']:
                                    if col not in weekly_df.columns:
                                        # Look for similar column names
                                        matches = [c for c in weekly_df.columns if c.lower() == col.lower()]
                                        if matches:
                                            weekly_df[col] = weekly_df[matches[0]]
                                
                                self.weekly_data[symbol] = weekly_df
                            
                            if daily_df is not None:
                                # Ensure column names are standardized
                                for col in ['Open', 'High', 'Low', 'Close']:
                                    if col not in daily_df.columns:
                                        # Look for similar column names
                                        matches = [c for c in daily_df.columns if c.lower() == col.lower()]
                                        if matches:
                                            daily_df[col] = daily_df[matches[0]]
                                
                                self.daily_data[symbol] = daily_df
                                
                        print(f"Processed {len(self.weekly_data)} weekly and {len(self.daily_data)} daily datasets")
                
                # Use our custom subclass
                scorer = UploadedDataMomentumScorer(symbols=symbols_from_files)
            else:
                print("No symbol data found in uploaded files, using default symbols")
                scorer = MarketMomentumScorer()
        else:
            print("No uploaded files found, using default symbols")
            scorer = MarketMomentumScorer()
        
        # Continue with the standard flow
        scorer.fetch_data()
        scorer.compute_indicators()
        results_df = scorer.build_summary_table()
    
        # Create the HTML table using Plotly
        def format_ma_touch(val):
            return "✔️" if val else "✖️"
    
        def get_color(val):
            return {
                "green": "#c6efce",
                "yellow": "#ffeb9c",
                "red": "#ffc7ce",
                "gray": "#eeeeee"  # Add gray for missing data
            }.get(val, "#eeeeee")
    
        momentum_weekly_colors = results_df["momentum_color_weekly"].map(get_color)
        momentum_daily_colors = results_df["momentum_color_daily"].map(get_color)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Symbol", "Weekly Momentum", "Touched MA (Weekly)", "Daily Momentum", "Touched MA (Daily)"],
                fill_color="lightgray",
                align="center",
                font=dict(size=14)
            ),
            cells=dict(
                values=[
                    results_df["symbol"],
                    results_df["momentum_color_weekly"],
                    results_df["touch_recent_ma_weekly"].map(format_ma_touch),
                    results_df["momentum_color_daily"],
                    results_df["touch_recent_ma_daily"].map(format_ma_touch)
                ],
                fill_color=[
                    ["white"] * len(results_df),  # Symbol
                    momentum_weekly_colors,
                    ["#f9f9f9"] * len(results_df),
                    momentum_daily_colors,
                    ["#f9f9f9"] * len(results_df)
                ],
                align="center",
                font=dict(size=12),
                height=28
            )
        )])
        
        # Add source information to the title
        data_source = "Using Your Uploaded Data" if have_uploaded_files and symbols_from_files else "Using External Data"
        fig.update_layout(
            title_text=f"Momentum Table (Weekly + Daily) - {data_source}", 
            margin=dict(t=50, b=20)
        )
        
        # Enhanced HTML with data source information
        html_page = f"""
        <html>
        <head>
        <title>Momentum Tables</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body style="font-family: Arial; margin: 40px;">
            <div class="container">
            <h1>📊 Momentum Summary</h1>
            
            <div class="alert {'alert-success' if have_uploaded_files and symbols_from_files else 'alert-info'}">
                <strong>Data Source:</strong> {data_source}
                {f'<br>Symbols analyzed: {", ".join(symbols_from_files)}' if have_uploaded_files and symbols_from_files else ''}
                {f'<br>Files used: {", ".join(symbol_data.keys())}' if have_uploaded_files else ''}
                </div>
            
            {fig.to_html(full_html=False)}
            
            <div class="mt-3">
                <a href="/" class="btn btn-primary">Return to Dashboard</a>
            </div>
            </div>
        </body>
        </html>
        """
        
        # Save results for dashboard display
        daily_results = [{
            "filename": "📊 Momentum Analysis",
            "text": f"Analysis completed for {len(results_df)} symbols.\nWeekly and daily trends calculated with SMA crossover detection.\n\nClick for full report."
        }]
        
        # Return the full HTML page with plots
        return html_page
        
    except Exception as e:
        error_details = traceback.format_exc()
        error_message = f"Error in momentum analysis: {str(e)}\n\nStacktrace:\n{error_details}"
        print(error_message)  # Log to console/logs
        
        # Add error to daily_results
        daily_results = [{
            "filename": "❌ Error in Momentum Analysis",
            "text": error_message
        }]
        
        # Return both HTML display and JSON error
        return render_template_string("""
        <html>
        <head>
            <title>Error in Momentum Analysis</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body style="font-family: Arial; margin: 40px;">
            <div class="container">
                <div class="alert alert-danger">
                    <h3>Error in Momentum Analysis</h3>
                    <p>{{ error }}</p>
                    <pre>{{ traceback }}</pre>
                </div>
                <a href="/" class="btn btn-primary">Return to Dashboard</a>
            </div>
        </body>
        </html>
        """, error=str(e), traceback=error_details)

@app.route("/image_analysis", methods=["POST"])
def image_analysis():
    print("===== IMAGE_ANALYSIS ROUTE CALLED =====")
    global image_results
    
    try:
        # Check for file upload from form
        if 'image_file' in request.files:
            file = request.files['image_file']
            if file.filename != '':
                # Process uploaded file
                file_bytes = file.read()
                
                # Analyze the image
                analysis = image_ai.analyze_image_with_bytes(file_bytes)
                if analysis and analysis.get('analysis'):
                    image_results = [{
                        "filename": f"📊 Chart Analysis: {file.filename}",
                        "text": analysis.get('analysis')
                    }]
                else:
                    image_results = [{
                        "filename": "❓ Chart Analysis",
                        "text": "No analysis was generated for the image. The image may not contain analyzable chart data."
                    }]
        else:
            # No file upload, try to use one of the other methods
            method = request.form.get('method', 'upload')
            
            if method == 'clipboard':
                # Process clipboard image
                if not image_ai.analyze_clipboard_snapshot():
                    image_results = [{
                        "filename": "❌ Clipboard Analysis Failed",
                        "text": "Could not capture clipboard data or analyze the image. Please ensure you have copied an image to your clipboard."
                    }]
            elif method == 'pick':
                # User wants to pick images
                results = image_ai.pick_snapshots_and_analyze()
                if results:
                    image_results = []
                    for filename, analysis in results.items():
                        image_results.append({
                            "filename": f"📊 Chart Analysis: {os.path.basename(filename)}",
                            "text": analysis.get('analysis', 'No analysis available')
                        })
                else:
                    image_results = [{
                        "filename": "❌ No Images Analyzed",
                        "text": "No images were selected or the analysis failed."
                    }]
            else:
                image_results = [{
                    "filename": "❓ Unknown Method",
                    "text": f"Unknown analysis method: {method}. Please try again."
                }]
        
        return redirect(url_for("index", no_reset="true"))
        
    except Exception as e:
        error_details = traceback.format_exc()
        error_message = f"Error in image analysis: {str(e)}\n\nStacktrace:\n{error_details}"
        print(error_message)  # Log to console/logs
        
        # Add error to image_results
        image_results = [{
            "filename": "❌ Error in Image Analysis",
            "text": error_message
        }]
        
        # Return both HTML display and JSON error
        return render_template_string("""
        <html>
        <head>
            <title>Error in Image Analysis</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body style="font-family: Arial; margin: 40px;">
            <div class="container">
                <div class="alert alert-danger">
                    <h3>Error in Image Analysis</h3>
                    <p>{{ error }}</p>
                    <pre>{{ traceback }}</pre>
                </div>
                <a href="/" class="btn btn-primary">Return to Dashboard</a>
            </div>
        </body>
        </html>
        """, error=str(e), traceback=error_details)

@app.route("/reset", methods=["POST"])
def reset():
    global daily_results, image_results, symbol_data
    
    # Clear in-memory data
    SUMMARY_CACHE.clear()
    EVENT_CACHE.clear()
    fetchers.token_usage = 0
    fetchers.cost_usd = 0
    daily_results = []
    image_results = []
    symbol_data.clear()
    app.config['FILE_INFO'] = []
    
    # Clear files from upload directories
    try:
        # Clear uploaded_csvs directory
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"Deleted: {file_path}")
                
        # Clear temp_uploads directory if it exists
        temp_dir = os.path.abspath("temp_uploads")
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error clearing files: {str(e)}")
    
    print("Reset: Cleared all data and files")
    return redirect(url_for("index"))

@app.route("/symbol_chart")
def symbol_chart():
    symbol_request = request.args.get("symbol")
    
    # Find matching file data in uploaded files
    chart_data = None
    file_used = None
    
    for filename, data_dict in symbol_data.items():
        if symbol_request.lower() in filename.lower():
            # Prefer daily data if available
            if "Daily" in data_dict:
                chart_data = data_dict["Daily"]
                file_used = f"{filename} (Daily)"
                break
            # Otherwise use the first available timeframe
            else:
                for timeframe, df in data_dict.items():
                    chart_data = df
                    file_used = f"{filename} ({timeframe})"
                    break
    
    # If no data found, return error message
    if chart_data is None:
        return f"""
        <html>
        <head><title>No Data Available</title></head>
        <body style="font-family:Arial; margin:40px">
            <h1>⚠️ No Data Available for {symbol_request}</h1>
            <p>Please upload data files for this symbol.</p>
            <a href="/" class="btn btn-primary">Return to Dashboard</a>
        </body>
        </html>
        """
    
    # Ensure required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in chart_data.columns:
            # Try to find case-insensitive matches
            matches = [c for c in chart_data.columns if c.lower() == col.lower()]
            if matches:
                chart_data[col] = chart_data[matches[0]]
            else:
                return f"""
                <html>
                <head><title>Missing Data Columns</title></head>
                <body style="font-family:Arial; margin:40px">
                    <h1>⚠️ Missing Required Column: {col}</h1>
                    <p>The data for {symbol_request} must include: Open, High, Low, Close</p>
                    <a href="/" class="btn btn-primary">Return to Dashboard</a>
                </body>
                </html>
                """
    
    # Ensure we have a date column or index
    if not isinstance(chart_data.index, pd.DatetimeIndex):
        date_columns = [col for col in chart_data.columns if any(date_term in col.lower() 
                                                        for date_term in ['date', 'time', 'datetime'])]
        if date_columns:
            date_col = date_columns[0]
            chart_data[date_col] = pd.to_datetime(chart_data[date_col])
            chart_data.set_index(date_col, inplace=True)
    
    # Calculate indicators
    chart_data["SMA_10"] = ta.sma(chart_data["Close"], length=10)
    chart_data["SMA_20"] = ta.sma(chart_data["Close"], length=20)

    def momentum_color(row):
        sma10 = row.get("SMA_10")
        sma20 = row.get("SMA_20")
        close = row.get("Close")

        if pd.isna(sma10) or pd.isna(sma20) or pd.isna(close):
            return "gray"

        if sma10 > sma20 and close > sma20:
            return "green"
        elif sma10 > sma20 and close <= sma20:
            return "yellow"
        elif sma10 < sma20 and close > sma20:
            return "yellow"
        else:
            return "red"

    chart_data["color"] = chart_data.apply(momentum_color, axis=1)

    fig = go.Figure()

    # Add candles per color
    for color in ["green", "yellow", "red"]:
        subset = chart_data[chart_data["color"] == color]
        if not subset.empty:
            fig.add_trace(go.Candlestick(
                x=subset.index,
                open=subset["Open"],
                high=subset["High"],
                low=subset["Low"],
                close=subset["Close"],
                name=color,
                increasing_line_color="green" if color == "green" else
                                      "gold" if color == "yellow" else "red",
                decreasing_line_color="green" if color == "green" else
                                      "gold" if color == "yellow" else "red",
                showlegend=True
            ))

    # Add MAs
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["SMA_10"], name="SMA 10", line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["SMA_20"], name="SMA 20", line=dict(color="black", width=1)))

    fig.update_layout(title=f"{symbol_request} Momentum Chart (Using Uploaded Data)", xaxis_title="Date", yaxis_title="Price")

    return f"""
    <html>
    <head>
        <title>{symbol_request} Chart</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body style="font-family:Arial; margin:40px">
        <div class="container">
            <h1>📈 {symbol_request} Momentum Chart</h1>
            <div class="alert alert-info">
                <strong>Data Source:</strong> {file_used}
            </div>
            {fig.to_html(full_html=False)}
            <div class="mt-3">
                <a href="/" class="btn btn-primary">Return to Dashboard</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.route("/test")
def test_route():
    return "Test route is working!"

if __name__ == "__main__":
    import os
    import sys
    
    # For PyCharm debugging compatibility
    if os.environ.get("PYCHARM_DEBUG") == "1":
        # Use a completely direct approach that bypasses Flask's run method
        from werkzeug.serving import run_simple
        run_simple("127.0.0.1", 8080, app, use_reloader=False, use_debugger=False)
    else:
        # For production or normal local run
        port = int(os.environ.get("PORT", 8080))
        app.run(host="0.0.0.0", port=port)


