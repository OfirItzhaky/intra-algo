# Import the numpy patch to fix NaN issue

from flask import Flask, request, render_template_string, redirect, url_for,jsonify, render_template, session, send_from_directory
from datetime import datetime
import plotly.graph_objects as go
import pandas_ta as ta

from research_agent.research_fetchers import ResearchFetchers, summarize_with_cache, summarize_economic_events_with_cache
from research_agent.research_analyzer import ResearchAnalyzer
from research_agent.news_aggregator import NewsAggregator
import os
import pandas as pd

from research_agent.scalp_agent.multitimeframe3strategies_agent import MultiTimeframe3StrategiesAgent
from research_agent.templates import HTML_TEMPLATE
import traceback
from research_agent.scalp_agent.scalp_agent_session import ScalpAgentSession
from research_agent.scalp_agent.scalp_agent_controller import ScalpAgentController
from research_agent.scalp_agent.input_container import InputContainer
from research_agent.scalp_agent.agent_handler import AgentHandler
from research_agent.scalp_agent.instinct_agent import InstinctAgent
from research_agent.scalp_agent.playbook_simulator import PlaybookSimulator
from research_agent.scalp_agent.csv_utils import validate_csv_against_indicators
from research_agent.scalp_agent.regression_prediction_agent import RegressionPredictorAgent
import numpy as np
import time

from research_agent.config import CONFIG, SUMMARY_CACHE, EVENT_CACHE, REGRESSION_STRATEGY_DEFAULTS


# === Runtime Constants ===
today = datetime.today().strftime("%Y-%m-%d")
DEFAULT_MARKETS = ["US"]
DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "NVDA"]
FOCUS_SECTORS = ["Technology", "Healthcare", "Energy"]
ONLY_MAJOR_EVENTS = False
PRINT_LIVE_SUMMARY = True
COPY_TO_CLIPBOARD = False
SAVE_DIRECTORY = "research_outputs"


# --- Global progress tracker ---
regression_backtest_tracker = {
    "current": 0,
    "total": 0,
    "start_time": None,
    "status": "idle",
    "cancel_requested": False
}

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

# Five Star Agent uploads folder
FIVESTAR_UPLOAD_FOLDER = os.path.abspath("uploaded_5star_images")
os.makedirs(FIVESTAR_UPLOAD_FOLDER, exist_ok=True)

# In-memory server-side store for Five Star Agent session data to avoid
# storing large blobs in client-side cookies.
# Structure: { session_id: { 'images': List[str], 'image_summary': Optional[str] } }
FIVESTAR_STORE: dict = {}

from uuid import uuid4

# --- Simple username login (case-insensitive) ---
# Build allow-list from env or fallback to defaults so the gate is active by default
_users_raw = os.getenv("RESEARCH_AGENT_USERS")
if not _users_raw or not _users_raw.strip():
    _users_raw = "ofir,itz01"
ALLOWED_USERS = [u.strip().lower() for u in _users_raw.split(",") if u.strip()]

# Force re-login on server restarts
SERVER_BOOT_ID = uuid4().hex

LOGIN_TEMPLATE = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Login</title></head><body style="font-family:system-ui,Segoe UI,Arial,sans-serif; padding:40px;">
  <h2>Enter your username:</h2>
  {% if error %}<div style="color:#b00020; font-weight:600; margin:8px 0 16px;">{{ error }}</div>{% endif %}
  <form method="post" action="{{ url_for('login') }}">
    <input name="username" autofocus style="padding:8px 12px; font-size:16px;" />
    <button type="submit" style="padding:8px 14px; font-size:16px; margin-left:8px;">Continue</button>
  </form>
</body></html>
"""

@app.before_request
def enforce_login():
    # Allow static and login endpoints
    endpoint = request.endpoint or ""
    if endpoint in {"login", "static"}:
        return
    # If no allow-list configured, skip gate
    if not ALLOWED_USERS:
        return
    # Check session
    username = session.get("username")
    boot_id = session.get("boot_id")
    if username and username.strip().lower() in ALLOWED_USERS and boot_id == SERVER_BOOT_ID:
        return
    # Block anything else until login; preserve target for post-login redirect
    if request.method == "GET":
        login_url = url_for("login", next=request.path)
        return redirect(login_url)
    # For POSTs, redirect to login page
    return redirect(url_for("login", next=request.path))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        entered = (request.form.get("username") or "").strip().lower()
        if entered and (entered in ALLOWED_USERS):
            session["username"] = entered
            session.permanent = False

            session["boot_id"] = SERVER_BOOT_ID
            # Redirect to originally requested path if present
            next_url = request.args.get("next") or url_for("index")
            return redirect(next_url)
        return render_template_string(LOGIN_TEMPLATE, error="Access Denied: You are not authorized to use this tool.")
    return render_template_string(LOGIN_TEMPLATE)

def _get_session_id():
    """Return a short, stable server-side session id for FiveStar.

    We avoid using the large Flask signed cookie; instead, we keep a tiny
    identifier in `session['fivestar_sid']` and store heavy data in the
    in-process FIVESTAR_STORE.
    """
    try:
        sid = session.get('fivestar_sid')
        if not sid:
            sid = uuid4().hex
            session['fivestar_sid'] = sid
            print(f"[FiveStar][DEBUG] Issued new fivestar_sid={sid}")
        return sid
    except Exception:
        return 'fivestar'

def _get_store_entry(session_id: str) -> dict:
    entry = FIVESTAR_STORE.get(session_id)
    if entry is None:
        entry = {'images': [], 'image_summary': None, 'chat': []}
        FIVESTAR_STORE[session_id] = entry
    return entry

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

# Utility functions for agent memory

def get_instinct_memory() -> dict:
    """
    Retrieve the last InstinctAgent result from session memory.
    Returns:
        dict: Last InstinctAgent result or empty dict.
    """
    return session.get("instinct_results", {})

def set_instinct_memory(result: dict) -> None:
    """
    Store the InstinctAgent result in session memory.
    Args:
        result (dict): InstinctAgent result to store.
    """
    session["instinct_results"] = result

def get_playbook_memory() -> dict:
    """
    Retrieve the last PlaybookAgent result from session memory.
    Returns:
        dict: Last PlaybookAgent result or empty dict.
    """
    return session.get("playbook_results", {})

def set_playbook_memory(result: dict) -> None:
    """
    Store the PlaybookAgent result in session memory.
    Args:
        result (dict): PlaybookAgent result to store.
    """
    session["playbook_results"] = result

# --- Session-level LLM cost tracking ---
def update_llm_session_cost(token_usage, cost_usd, model_name):
    if 'llm_session_total_tokens' not in session:
        session['llm_session_total_tokens'] = 0
    if 'llm_session_total_cost' not in session:
        session['llm_session_total_cost'] = 0.0
    if 'llm_model_name' not in session:
        session['llm_model_name'] = model_name
    # Parse token_usage (e.g., "Prompt: 765 × 3, Output: ~250 × 3")
    import re
    tokens = 0
    if isinstance(token_usage, str):
        matches = re.findall(r'(\d+)', token_usage)
        tokens = sum(int(m) for m in matches)
    elif isinstance(token_usage, int):
        tokens = token_usage
    session['llm_session_total_tokens'] += tokens
    session['llm_session_total_cost'] += float(cost_usd or 0.0)
    session['llm_model_name'] = model_name

def get_llm_session_cost():
    return {
        'llm_session_total_tokens': session.get('llm_session_total_tokens', 0),
        'llm_session_total_cost': session.get('llm_session_total_cost', 0.0),
        'llm_model_name': session.get('llm_model_name', 'Gemini 1.5 Pro')
    }

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
                
                # --- Normalize volume column to 'volume' ---
                vol_col = next((col for col in df.columns if 'vol' in col.lower()), None)
                if vol_col and vol_col != 'volume':
                    df.rename(columns={vol_col: 'volume'}, inplace=True)
                
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
        upload_error=upload_error,
        scalp_agent_url=url_for('scalp_agent'),
        five_star_agent_url=url_for('five_star_agent')
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
        return render_template_string(
        """
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
            return render_template_string(
            """
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

        # Only use UploadedDataMomentumScorer
        from market_momentum_scorer import MarketMomentumScorer

        symbols_from_files = []
        for filename in symbol_data.keys():
            base_name = os.path.splitext(filename)[0].lower()
            if "_daily" in base_name:
                symbol = base_name.replace("_daily", "").upper()
                if symbol not in symbols_from_files:
                    symbols_from_files.append(symbol)
            elif "_weekly" in base_name:
                symbol = base_name.replace("_weekly", "").upper()
                if symbol not in symbols_from_files:
                    symbols_from_files.append(symbol)

        if not symbols_from_files:
            return render_template_string(
            """
            <html>
            <head>
                <title>No Valid Symbols</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body style="font-family: Arial; margin: 40px;">
                <div class="container">
                    <div class="alert alert-warning">
                        <h3>No Valid Symbols Found</h3>
                        <p>Your uploaded files did not contain recognizable symbols. Please check your files and try again.</p>
                    </div>
                    <a href="/" class="btn btn-primary">Return to Dashboard</a>
                </div>
            </body>
            </html>
            """)

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
                
        scorer = UploadedDataMomentumScorer(symbols=symbols_from_files)
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
        data_source = "Using Your Uploaded Data"
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
            
            <div class="alert alert-success">
                <strong>Data Source:</strong> {data_source}
                {f'<br>Symbols analyzed: {", ".join(symbols_from_files)}' if symbols_from_files else ''}
                {f'<br>Files used: {", ".join(symbol_data.keys())}' if symbols_from_files else ''}
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
        return render_template_string(
        """
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

@app.route("/scalp-agent", methods=["GET", "POST"])
def scalp_agent():
    import pandas as pd
    csv_request_info = None
    validation_result = None
    result = None
    show_csv_upload = False

    if request.method == "POST":
        # Phase 1: Chart image upload and requirements extraction
        if 'chart_file' in request.files and request.files['chart_file'].filename:
            chart_file = request.files['chart_file']
            session_notes = request.form.get("session_notes")
            image_bytes = chart_file.read()
            from research_agent.scalp_agent.scalp_agent_session import ScalpAgentSession
            session_obj = ScalpAgentSession(image_bytes=image_bytes, session_notes=session_notes)
            csv_request_info = session_obj.analyze_chart_image()
            # Store requirements in Flask session for next phase
            session['scalp_agent_requirements'] = csv_request_info
            # --- Gemini token usage/cost info ---
            gemini_cost_info = None
            usage = getattr(session_obj, 'token_usage_summary', None)
            if usage:
                total_tokens = usage.get('totalTokenCount', 'N/A')
                try:
                    cost = float(total_tokens) * 0.007 / 1000 if total_tokens != 'N/A' else 'N/A'
                except Exception:
                    cost = 'N/A'
                gemini_cost_info = {
                    'promptTokenCount': usage.get('promptTokenCount', 'N/A'),
                    'candidatesTokenCount': usage.get('candidatesTokenCount', 'N/A'),
                    'totalTokenCount': total_tokens,
                    'cost_estimate': f"${cost:.4f}" if isinstance(cost, float) else 'N/A'
                }
            # --- Human-readable requirements summary ---
            requirements_summary = session_obj.get_requirements_summary()
            result = {
                "max_risk": request.form.get("max_risk"),
                "max_risk_per_trade": request.form.get("max_risk_per_trade"),
                "notes": session_notes,
                "csv_filename": None,  # Not processed yet
                "chart_filename": chart_file.filename if chart_file else None
            }
            show_csv_upload = True
            return render_template(
                "scalp_agent.html",
                result=result,
                csv_request_info=csv_request_info,
                show_csv_upload=show_csv_upload,
                validation_result=None,
                trade_idea_result=None,
                max_risk_per_trade=request.form.get("max_risk_per_trade"),
                gemini_cost_info=gemini_cost_info,
                requirements_summary=requirements_summary,
                csv_summaries=[]
            )
        # Phase 2: CSV upload and validation (optional)
        elif 'csv_files' in request.files or (hasattr(request.files, 'getlist') and request.files.getlist('csv_files')):
            csv_files = request.files.getlist('csv_files')
            csv_summaries = []
            for csv_file in csv_files:
                if csv_file and csv_file.filename:
                    try:
                        df = pd.read_csv(csv_file)
                        num_bars = len(df)
                        fname = csv_file.filename
                        tf = "1m" if "1min" in fname.lower() else "5m"
                        # Optionally, smarter inference can be added here
                        csv_summaries.append({
                            'filename': fname,
                            'num_bars': num_bars,
                            'timeframe': tf
                        })
                    except Exception as e:
                        csv_summaries.append({
                            'filename': csv_file.filename,
                            'num_bars': 0,
                            'timeframe': 'Unknown',
                            'error': str(e)
                        })
            requirements = session.get('scalp_agent_requirements')
            return render_template(
                "scalp_agent.html",
                result=None,
                csv_request_info=requirements,
                show_csv_upload=True,
                validation_result=None,
                trade_idea_result=None,
                max_risk_per_trade=request.form.get("max_risk_per_trade"),
                csv_summaries=csv_summaries
            )
    # GET or initial load
    return render_template("scalp_agent.html", result=None, csv_request_info=None, show_csv_upload=False, validation_result=None, trade_idea_result=None, max_risk_per_trade=None)


# =============================
# Five Star Agent (Swing) Routes
# =============================

def _get_fivestar_chat_history():
    try:
        sid = _get_session_id()
        return list(_get_store_entry(sid).get('chat', []))
    except Exception:
        return []


from typing import Optional, List
import importlib


def _append_fivestar_message(role: str, content: str, images: Optional[List[str]] = None):
    from datetime import datetime as _dt
    sid = _get_session_id()
    entry = _get_store_entry(sid)
    chat = entry.get('chat', [])
    # Cap content for UI to prevent runaway mem (store shortened for UI only)
    max_ui_chars = 8000
    chat.append({
        'role': role,
        'content': (content or '')[:max_ui_chars],
        'images': images or [],
        'timestamp': _dt.utcnow().isoformat() + 'Z'
    })
    entry['chat'] = chat
    FIVESTAR_STORE[sid] = entry


@app.route("/five-star-agent", methods=["GET", "POST"])
def five_star_agent():
    """Five Star Agent placeholder UI and backend handler.

    - GET: Render chat UI with history
    - POST: Accept images + instructions, store to session, return placeholder response
    """
    if request.method == 'POST':
        instructions = request.form.get('instructions', '')
        model_choice = request.form.get('model_choice', '')
        uploaded_files = request.files.getlist('images') if 'images' in request.files or hasattr(request.files, 'getlist') else []

        saved_images = []
        for f in uploaded_files:
            if not f or not getattr(f, 'filename', ''):
                continue
            filename = f.filename
            # Only accept images
            lower = filename.lower()
            if not (lower.endswith('.png') or lower.endswith('.jpg') or lower.endswith('.jpeg')):
                continue
            save_path = os.path.join(FIVESTAR_UPLOAD_FOLDER, filename)
            # Ensure unique filename to avoid overwrite
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(save_path):
                filename = f"{base}_{counter}{ext}"
                save_path = os.path.join(FIVESTAR_UPLOAD_FOLDER, filename)
                counter += 1
            f.save(save_path)
            saved_images.append({'filename': filename, 'path': save_path})

        # Append user message
        _append_fivestar_message('user', instructions, images=[img['filename'] for img in saved_images])

        # Local placeholder (UI-only path; real calls use /five_star/analyze)
        received = ", ".join([img['filename'] for img in saved_images]) or "(no images)"
        agent_reply = (
            f"✅ Received charts: {received}\n"
            f"📝 Placeholder reply (use the Send to Agent button to analyze with model).\n"
            f"📌 Instructions acknowledged: {instructions[:140] + ('...' if len(instructions) > 140 else '') if instructions else ''}"
        )
        _append_fivestar_message('agent', agent_reply)

    chat_history = _get_fivestar_chat_history()
    # Active user for model gating
    active_user = (session.get("username") or os.getenv("RESEARCH_AGENT_ACTIVE_USER") or "").strip().lower()
    return render_template("5star_agent.html", chat_history=chat_history, active_user=active_user)


@app.route("/five_star/analyze", methods=["POST"])
def five_star_analyze():
    """API endpoint to analyze uploaded charts + instructions via LLM.

    Saves images, appends the user message to session chat, calls the Five Star
    Agent controller (OpenAI-backed), appends the agent message, and returns
    JSON with success/error.
    """
    print("[FiveStar] /five_star/analyze called")
    # Dynamically import controller (support both folder names: five_star_agent / 5_star_agent)
    import importlib.util as _ilu
    base_dir = os.path.dirname(__file__)
    candidate_paths = [
        os.path.join(base_dir, "five_star_agent", "five_star_agent_controller.py"),
        os.path.join(base_dir, "5_star_agent", "five_star_agent_controller.py"),
    ]
    module = None
    chosen_path = None
    for controller_path in candidate_paths:
        if os.path.exists(controller_path):
            chosen_path = controller_path
            spec = _ilu.spec_from_file_location("fivestar_controller", controller_path)
            module = _ilu.module_from_spec(spec)
            assert spec and spec.loader, "Failed to load FiveStarAgentController spec"
            spec.loader.exec_module(module)
            break
    if module is None:
        print("[FiveStar][ERROR] Controller not found in:", candidate_paths)
        return jsonify({"ok": False, "error": "FiveStarAgent controller not found."}), 500
    FiveStarAgentController = getattr(module, "FiveStarAgentController", None)
    if FiveStarAgentController is None:
        print(f"[FiveStar][ERROR] FiveStarAgentController class missing in {chosen_path}")
        return jsonify({"ok": False, "error": "Controller class missing."}), 500

    try:
        instructions = request.form.get('instructions', '')
        model_choice = request.form.get('model_choice', '')
        print(f"[FiveStar] model_choice={model_choice!r}")
        files = request.files.getlist('images') if 'images' in request.files or hasattr(request.files, 'getlist') else []
        print(f"[FiveStar] received {len(files)} files")

        saved_filepaths = []
        saved_names = []
        for f in files:
            if not f or not getattr(f, 'filename', ''):
                continue
            filename = f.filename
            lower = filename.lower()
            if not (lower.endswith('.png') or lower.endswith('.jpg') or lower.endswith('.jpeg')):
                continue
            save_path = os.path.join(FIVESTAR_UPLOAD_FOLDER, filename)
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(save_path):
                filename = f"{base}_{counter}{ext}"
                save_path = os.path.join(FIVESTAR_UPLOAD_FOLDER, filename)
                counter += 1
            f.save(save_path)
            saved_filepaths.append(save_path)
            saved_names.append(filename)
        print(f"[FiveStar][IMAGES] New uploaded files: {saved_filepaths}")

        # Determine prior session paths (if any) from server store, and sanitize
        sid = _get_session_id()
        store_entry = _get_store_entry(sid)
        prior_filepaths = store_entry.get('images', []) or []
        prior_filepaths = [p for p in prior_filepaths if os.path.exists(p)]
        prior_names = [os.path.basename(p) for p in prior_filepaths]

        # If no new images this turn, reuse prior (already sanitized)
        reused_from_session = False
        if len(saved_filepaths) == 0 and prior_filepaths:
            print(f"[FiveStar] no new images uploaded; reusing prior {len(prior_filepaths)} images")
            saved_filepaths = prior_filepaths
            saved_names = prior_names
            reused_from_session = True
            print(f"[FiveStar][IMAGES] Reused from session: {prior_filepaths}")

        # Final sanitize for current set (only store/keep paths that exist)
        saved_filepaths = [p for p in saved_filepaths if os.path.exists(p)]
        saved_names = [os.path.basename(p) for p in saved_filepaths]
        print(f"[FiveStar][IMAGES] Sent to LLM: {saved_filepaths}")

        # If still none, return helpful error
        if len(saved_filepaths) == 0:
            msg = "Please paste or upload at least one weekly chart image to analyze."
            print(f"[FiveStar][WARN] {msg}")
            return jsonify({"ok": False, "error": msg, "code": "NO_IMAGES"}), 400

        # Append user message to chat history only if we can proceed
        # Suppress reused filenames in the UI (they are still used internally)
        _append_fivestar_message('user', instructions, images=([] if reused_from_session else saved_names))

        # Override session image context with only the current, existing paths
        # Update server-side store (not client cookie)
        store_entry['images'] = list(saved_filepaths)
        FIVESTAR_STORE[sid] = store_entry
        print(f"[FiveStar] server store images set: {store_entry['images']}")

        # Call controller with LLM
        try:
            controller = FiveStarAgentController()
        except Exception as e:
            print(f"[FiveStar][ERROR] Failed to init controller: {e}")
            return jsonify({"ok": False, "error": f"Failed to init controller: {e}"}), 500
        try:
            session_id = sid
            # Build image summary injection policy
            image_summary = store_entry.get('image_summary')
            inject_summary = False
            if reused_from_session and image_summary:
                inject_summary = True
                print(f"[FiveStar][OPT] Reusing summary instead of images. words={len(str(image_summary).split())}")
            elif reused_from_session and not image_summary:
                print("[FiveStar][OPT] No summary available; images will be sent to preserve context.")
            if model_choice:
                agent_reply, model_used, usage = controller.analyze_with_model(
                    instructions=instructions,
                    image_paths=([] if inject_summary else saved_filepaths),
                    model_choice=model_choice,
                    session_id=session_id,
                    image_summary=(image_summary if inject_summary else None),
                    active_user=(session.get("username") or os.getenv("RESEARCH_AGENT_ACTIVE_USER") or "").strip().lower()
                )
            else:
                agent_reply, model_used, usage = controller.analyze_with_model(
                    instructions=instructions,
                    image_paths=([] if inject_summary else saved_filepaths),
                    model_choice="gpt-4o-mini",
                    session_id=session_id,
                    image_summary=(image_summary if inject_summary else None),
                    active_user=(session.get("username") or os.getenv("RESEARCH_AGENT_ACTIVE_USER") or "").strip().lower()
                )

            # If we reused images, strip the explicit filename echo from the top of reply
            if reused_from_session and isinstance(agent_reply, str) and agent_reply.startswith("✅ Received charts:"):
                try:
                    agent_reply = "\n".join(agent_reply.splitlines()[1:])
                except Exception:
                    pass

            # If this is the first turn with actual images, cache a summary for future follow-ups
            if not reused_from_session and len(saved_filepaths) > 0:
                # Store a capped summary server-side to control size
                max_chars = 8000
                store_entry['image_summary'] = agent_reply[:max_chars]
                FIVESTAR_STORE[sid] = store_entry
                try:
                    print(f"[FiveStar][OPT] Cached summary (capped) from first image turn. words={len(agent_reply.split())} chars={len(agent_reply)} -> stored_chars={len(store_entry['image_summary'])}")
                except Exception:
                    pass
        except Exception as e:
            provider = 'Gemini' if (model_choice or '').lower().startswith('gemini') else 'OpenAI'
            import traceback as _tb
            print(f"[FiveStar][ERROR] analyze_with_model failed ({provider}): {e}")
            print(f"[FiveStar][TRACE] {_tb.format_exc()}")
            return jsonify({"ok": False, "error": f"{provider} inference failed: {e}"}), 500
        # Append to chat including model used (already appended in reply, but keep metadata minimal)
        _append_fivestar_message('agent', agent_reply)

        print(f"[FiveStar] Model used: {model_used}")
        # Log response sizes; session cookie should remain small now that data is server-side
        try:
            import json as _json
            payload_bytes = len(_json.dumps({"agent_reply": agent_reply, "usage": usage}, ensure_ascii=False).encode('utf-8'))
            print(f"[FiveStar][DEBUG] Response payload (partial) size: {payload_bytes} bytes")
            # We no longer store large data in Flask client-side session
            print(f"[FiveStar][DEBUG] Session cookie remains minimal; large data kept server-side for sid={sid}")
        except Exception as _e:
            print(f"[FiveStar][WARN] Size logging failed: {_e}")
        return jsonify({
            "ok": True,
            "agent_reply": agent_reply,
            "model_used": model_used,
            "usage": usage
        })
    except Exception as e:
        import traceback as _tb
        print(f"[FiveStar][ERROR] Unexpected: {e}")
        print(f"[FiveStar][TRACE] {_tb.format_exc()}")
        return jsonify({"ok": False, "error": str(e)}), 500


import os
from flask import session, redirect

@app.route('/five_star/reset', methods=['POST'])
def reset_five_star_session():
    print("[FiveStar][RESET] Clearing session memory and uploaded images")
    try:
        print(f"[FiveStar][RESET] Current session paths: {session.get('five_star_agent_images', [])}")
    except Exception:
        pass

    # Remove saved image files
    sid = _get_session_id()
    image_paths = []
    try:
        image_paths = list(_get_store_entry(sid).get('images', []))
    except Exception:
        pass
    for path in image_paths:
        try:
            os.remove(path)
            print(f"[FiveStar][RESET] Deleted file: {path}")
        except Exception as e:
            print(f"[FiveStar][RESET] Failed to delete file {path}: {e}")

    # Clear chat memory and server-side store
    session.pop('five_star_agent_chat', None)
    try:
        if sid in FIVESTAR_STORE:
            FIVESTAR_STORE[sid] = {'images': [], 'image_summary': None, 'chat': []}
    except Exception:
        pass

    # After reset, send user back with a confirmation flag
    return redirect('/five-star-agent?reset=1')


@app.route("/dual_agent_scalp_analysis", methods=["POST"])
def dual_agent_scalp_analysis():
    """
    Accepts a chart image, analyzes it, builds an InputContainer, runs dual-agent analysis,
    and returns required indicators and agent outputs as JSON.
    """
    if 'chart_file' not in request.files or request.files['chart_file'].filename == '':
        return jsonify({"error": "No chart image uploaded."}),

    chart_file = request.files['chart_file']
    image_bytes = chart_file.read()
    session_notes = request.form.get("session_notes")
    max_risk = request.form.get("max_risk")
    max_risk_per_trade = request.form.get("max_risk_per_trade")

    # Analyze chart image to extract vision output
    session_obj = ScalpAgentSession(image_bytes=image_bytes, session_notes=session_notes)
    vision_output = session_obj.analyze_chart_image()

    # Build InputContainer from vision output and user input
    input_container = InputContainer(
        symbol=vision_output.get('symbol', ''),
        interval=vision_output.get('interval', ''),
        last_bar_time=vision_output.get('last_bar_time'),
        patterns=vision_output.get('patterns', []),
        indicators=vision_output.get('indicators', []),
        support_resistance_zones=vision_output.get('support_resistance_zones', []),
        rag_insights=vision_output.get('rag_insights'),
        user_notes=session_notes
    )
    # Prepare user_params
    user_params = {
        "max_risk": max_risk,
        "max_risk_per_trade": max_risk_per_trade,
        "session_notes": session_notes
    }
    # Run dual-agent analysis
    controller = ScalpAgentController()
    agent_results = controller.run_dual_agent_analysis(input_container, user_params)
    # Get all indicators needed
    agent_handler = AgentHandler()
    indicators_needed = agent_handler.get_all_indicators(input_container, user_params)
    # Prepare response
    return jsonify({
        "indicators_needed": indicators_needed,
        "instinct_agent": agent_results.get("InstinctAgent", {}),
        "playbook_agent": agent_results.get("PlaybookAgent", {})
    })

@app.route("/start_instinct", methods=["POST"])
def start_instinct():
    """
    Runs only InstinctAgent logic and returns the result for the left panel.
    """
    chart_file = request.files.get('chart_file')
    csv_file = request.files.get('csv_file')
    session_notes = request.form.get('session_notes')
    max_risk = request.form.get('max_risk')
    max_risk_per_trade = request.form.get('max_risk_per_trade')
    image_bytes = chart_file.read() if chart_file else None
    session_obj = ScalpAgentSession(image_bytes=image_bytes, session_notes=session_notes)
    vision_output = session_obj.analyze_chart_image() if image_bytes else {}
    input_container = InputContainer(
        symbol=vision_output.get('symbol', ''),
        interval=vision_output.get('interval', ''),
        last_bar_time=vision_output.get('last_bar_time'),
        patterns=vision_output.get('patterns', []),
        indicators=vision_output.get('indicators', []),
        support_resistance_zones=vision_output.get('support_resistance_zones', []),
        rag_insights=vision_output.get('rag_insights'),
        user_notes=session_notes
    )
    user_params = {
        "max_risk": max_risk,
        "max_risk_per_trade": max_risk_per_trade,
        "session_notes": session_notes
    }
    # CSV context for agent feedback
    if csv_file and csv_file.filename:
        import pandas as pd
        df = pd.read_csv(csv_file)
        user_params["csv_uploaded"] = True
        user_params["csv_length"] = len(df)
        user_params["csv_indicators_available"] = list(df.columns)
    else:
        user_params["csv_uploaded"] = False
    agent = InstinctAgent()
    result = agent.analyze(input_container, user_params)
    set_instinct_memory(result)

    # --- Ensure cost fields are present ---
    if "rag_token_usage" not in result:
        result["rag_token_usage"] = "N/A"
    if "rag_cost_usd" not in result:
        result["rag_cost_usd"] = 0.0
    if "llm_token_usage" not in result:
        result["llm_token_usage"] = "N/A"
    if "llm_cost_usd" not in result:
        result["llm_cost_usd"] = 0.0
    if "total_cost_usd" not in result:
        result["total_cost_usd"] = 0.0
    return jsonify(result)

@app.route("/start_playbook", methods=["POST"])
def start_playbook():
    """
    Runs PlaybookAgent with PlaybookSimulator and returns the result for the right panel.
    """
    chart_files = request.files.getlist('chart_files')
    csv_file = request.files.get('csv_file')
    session_notes = request.form.get('session_notes')
    max_risk = request.form.get('max_risk')
    max_risk_per_trade = request.form.get('max_risk_per_trade')
    # Collect tags for each image
    chart_file_tags = []
    for idx in range(len(chart_files)):
        tag = request.form.get(f'chart_file_tag_{idx}', '')
        chart_file_tags.append(tag)
    # If no images, fallback to single chart_file for backward compatibility
    if not chart_files and request.files.get('chart_file'):
        chart_files = [request.files.get('chart_file')]
        chart_file_tags = [request.form.get('chart_file_tag_0', '')]
    # Analyze all images and collect vision outputs
    vision_outputs = []
    for img_file, tag in zip(chart_files, chart_file_tags):
        if img_file:
            image_bytes = img_file.read()
            session_obj = ScalpAgentSession(image_bytes=image_bytes, session_notes=session_notes)
            vision_output = session_obj.analyze_chart_image() if image_bytes else {}
            vision_output['timeframe_tag'] = tag
            vision_outputs.append(vision_output)
    # Compose input_container from all vision outputs (pass the list)
    input_container = {
        'vision_outputs': vision_outputs,
        'user_notes': session_notes
    }
    user_params = {
        "max_risk": max_risk,
        "max_risk_per_trade": max_risk_per_trade,
        "session_notes": session_notes
    }
    agent = MultiTimeframe3StrategiesAgent()
    agent_result = agent.analyze(input_container, user_params)
    # If CSV is uploaded, validate and simulate
    simulation_result = None
    if csv_file and csv_file.filename:
        import pandas as pd
        df = pd.read_csv(csv_file)
        df.columns = [col.lower() for col in df.columns]
        required_indicators = agent_result.get('indicators', [])
        validation = validate_csv_against_indicators(df, required_indicators)
        if validation['is_valid']:
            simulator = PlaybookSimulator()
            # Simulate each strategy in the bank
            simulation_result = []
            for strat in agent_result.get('strategies', []):
                sim = simulator.simulate(df, strat)
                simulation_result.append({
                    'strategy_name': strat.get('strategy_name'),
                    'metrics': sim
                })
        else:
            simulation_result = {'csv_valid': False, 'missing': validation['missing']}
    playbook_results = {
        'playbook_strategies': agent_result.get('strategies', []),
        'csv_simulation': simulation_result
    }
    # --- Add bias_summary and LLM cost fields ---
    if 'bias_summary' in agent_result:
        playbook_results['bias_summary'] = agent_result['bias_summary']
    playbook_results['llm_token_usage'] = "Prompt: 765, Output: 235"
    playbook_results['llm_cost_usd'] = 0.0037
    playbook_results['total_cost_usd'] = 0.0037
    # HIGHLIGHTED: Add feedback if both image and CSV are missing
    if not image_bytes and not csv_file:
        playbook_results["feedback"] = "Please upload at least a chart image or CSV file to start strategy generation."
    # END HIGHLIGHTED
    set_playbook_memory(playbook_results)

    # --- Ensure cost fields are present ---
    if "rag_token_usage" not in playbook_results:
        playbook_results["rag_token_usage"] = "N/A"
    if "rag_cost_usd" not in playbook_results:
        playbook_results["rag_cost_usd"] = 0.0
    if "llm_token_usage" not in playbook_results:
        playbook_results["llm_token_usage"] = "N/A"
    if "llm_cost_usd" not in playbook_results:
        playbook_results["llm_cost_usd"] = 0.0
    return jsonify(playbook_results)

@app.route("/query_instinct", methods=["POST"])
def query_instinct():
    """
    Receives a follow-up question and returns a reply based on previous InstinctAgent analysis.
    Uses session memory for context.
    """
    data = request.get_json()
    question = data.get("question", "")
    instinct_memory = get_instinct_memory()
    # For now, just echo the question and show the last summary/strategies
    # (Replace with LLM or more advanced logic as needed)
    reply = {
        "question": question,
        "last_summary": instinct_memory.get("summary", "No previous summary."),
        "strategies": instinct_memory.get("strategies", [])
    }
    return jsonify(reply)

@app.route("/query_playbook", methods=["POST"])
def query_playbook():
    """
    Receives a follow-up question and returns a reply based on previous PlaybookAgent simulation results.
    Uses session memory for context.
    """
    data = request.get_json()
    question = data.get("question", "")
    playbook_memory = get_playbook_memory()
    # For now, just echo the question and show the last simulation results
    # (Replace with LLM or more advanced logic as needed)
    reply = {
        "question": question,
        "csv_simulation": playbook_memory.get("csv_simulation", None),
        "strategies": playbook_memory.get("playbook_strategies", [])
    }
    return jsonify(reply)

@app.route("/start_multitimeframe_agent", methods=["POST"])
def start_multitimeframe_agent():
    """
    Runs MultiTimeframe3StrategiesAgent on multiple chart images and returns bias summary and raw Gemini data.
    """
    chart_files = request.files.getlist('chart_files')
    session_notes = request.form.get('session_notes')
    chart_file_tags = []
    for idx in range(len(chart_files)):
        tag = request.form.get(f'chart_file_tag_{idx}', '')
        chart_file_tags.append(tag)
    # Fallback for single image
    if not chart_files and request.files.get('chart_file'):
        chart_files = [request.files.get('chart_file')]
        chart_file_tags = [request.form.get('chart_file_tag_0', '')]
    # If no images, return error
    if not chart_files or all([not f or not getattr(f, 'filename', None) for f in chart_files]):
        return jsonify({
            "bias_summary": [],
            "raw_bias_data": [],
            "feedback": "Please upload at least one chart image to run the MultiTimeframe Agent.",
            "step": "no_image"
        })
    # Analyze all images and collect vision outputs (attach image_bytes)
    vision_outputs = []
    for img_file, tag in zip(chart_files, chart_file_tags):
        if img_file:
            try:
                image_bytes = img_file.read()
                vision_outputs.append({
                    'image_bytes': image_bytes,
                    'timeframe_tag': tag
                })
            except Exception as e:
                vision_outputs.append({'error': f'Image analysis failed: {e}', 'timeframe_tag': tag})
    # Compose input_container from all vision outputs
    input_container = {
        'vision_outputs': vision_outputs,
        'user_notes': session_notes
    }
    user_params = {
        "session_notes": session_notes
    }
    agent = MultiTimeframe3StrategiesAgent()
    agent_result = agent.analyze(input_container, user_params)
    # Always include bias_summary and raw_bias_data
    model_name = "Gemini 1.5 Pro"
    bias_cost_metadata = agent_result.get("bias_cost_metadata", {})
    llm_token_usage = bias_cost_metadata.get("total_tokens")
    llm_cost_usd = bias_cost_metadata.get("total_cost_usd")
    update_llm_session_cost(llm_token_usage, llm_cost_usd, model_name)
    session_cost = get_llm_session_cost()
    result = {
        "bias_summary": agent_result.get("bias_summary", []),
        "raw_bias_data": agent_result.get("raw_bias_data", []),
        "feedback": agent_result.get("feedback", None),
        "llm_token_usage": llm_token_usage,
        "llm_cost_usd": llm_cost_usd,
        "total_cost_usd": llm_cost_usd,
        "step": agent_result.get("step", None),
        "llm_model_name": model_name,
        "llm_session_total_tokens": session_cost['llm_session_total_tokens'],
        "llm_session_total_cost": session_cost['llm_session_total_cost'],
        "llm_session_model_name": session_cost['llm_model_name'],
        "bias_cost_metadata": bias_cost_metadata
    }
    return jsonify(result)

@app.route("/run_regression_predictor", methods=["POST"])
def run_regression_predictor():
    print("[run_regression_predictor] Endpoint called.")
    import pandas as pd
    # Accept multiple files and select the 5m file by content
    files = request.files.getlist('csv_files')
    if not files or all(f.filename == '' for f in files):
        print("[run_regression_predictor] No CSV files uploaded.")
        return jsonify({'feedback': 'No CSV files uploaded. Please upload at least one CSV to run regression strategy.'})
    def infer_interval_from_df(df, filename=None):
        # Try to infer interval from the time column and date column
        time_col = None
        date_col = None
        for c in df.columns:
            if c.lower() in ['time', 'datetime', 'timestamp']:
                time_col = c
            if c.lower() == 'date':
                date_col = c
        print(f"[interval_detect] {filename}: columns={list(df.columns)}, time_col={time_col}, date_col={date_col}, n_rows={len(df)}")
        if not time_col:
            print(f"[interval_detect] {filename}: No time column found.")
            return None
        n = len(df)
        if n < 2:
            print(f"[interval_detect] {filename}: Not enough rows.")
            return None
        # Use last 5 bars if possible
        last_rows = df.iloc[-5:] if n >= 5 else df.iloc[-2:]
        # Check if all dates are the same
        if date_col and len(set(last_rows[date_col].astype(str))) == 1:
            # All same date, use all rows
            times = last_rows[time_col].astype(str).tolist()
            dates = last_rows[date_col].astype(str).tolist()
            print(f"[interval_detect] {filename}: Using last 5 times (same date): {times}, date: {dates[0] if dates else 'N/A'}")
        else:
            # Use only last 2 bars
            times = df.iloc[-2:][time_col].astype(str).tolist()
            dates = df.iloc[-2:][date_col].astype(str).tolist() if date_col else ['N/A', 'N/A']
            print(f"[interval_detect] {filename}: Using last 2 times (diff date): {times}, dates: {dates}")
        if len(times) < 2:
            print(f"[interval_detect] {filename}: Not enough times for diff.")
            return None
        import re
        mins = []
        for t in times:
            m = re.match(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', t)
            if m:
                h = int(m[1]); m2 = int(m[2]); s = int(m[3]) if m[3] else 0
                mins.append(h * 60 + m2 + s / 60)
            else:
                print(f"[interval_detect] {filename}: Could not parse time '{t}'")
        print(f"[interval_detect] {filename}: minute values: {mins}")
        if len(mins) < 2:
            print(f"[interval_detect] {filename}: Not enough valid minute values.")
            return None
        diffs = [mins[i+1] - mins[i] for i in range(len(mins)-1)]
        print(f"[interval_detect] {filename}: diffs: {diffs}")
        # Only consider positive diffs
        pos_diffs = [d for d in diffs if d > 0]
        print(f"[interval_detect] {filename}: positive diffs: {pos_diffs}")
        if not pos_diffs:
            print(f"[interval_detect] {filename}: No positive diffs.")
            return None
        # Use mode (most common) difference
        from collections import Counter
        mode_diff, _ = Counter(pos_diffs).most_common(1)[0]
        print(f"[interval_detect] {filename}: mode_diff: {mode_diff}")
        if abs(mode_diff - 1) < 0.1:
            print(f"[interval_detect] {filename}: Detected interval: 1m")
            return '1m'
        if abs(mode_diff - 5) < 0.1:
            print(f"[interval_detect] {filename}: Detected interval: 5m")
            return '5m'
        if abs(mode_diff - 15) < 0.1:
            print(f"[interval_detect] {filename}: Detected interval: 15m")
            return '15m'
        if abs(mode_diff - 30) < 0.1:
            print(f"[interval_detect] {filename}: Detected interval: 30m")
            return '30m'
        if abs(mode_diff - 60) < 1:
            print(f"[interval_detect] {filename}: Detected interval: 60m")
            return '60m'
        print(f"[interval_detect] {filename}: Detected interval: {mode_diff}m")
        return f'{mode_diff}m'
    file_5m = None
    for f in files:
        try:
            df = pd.read_csv(f)
            # --- Normalize columns and volume column ---
            df.columns = [col.lower() for col in df.columns]
            vol_col = next((col for col in df.columns if 'vol' in col), None)
            if vol_col:
                if vol_col != 'volume':
                    df.rename(columns={vol_col: 'volume'}, inplace=True)
            else:
                raise ValueError(f"No volume column found in uploaded file {f.filename}. Columns: {list(df.columns)}")
            interval = infer_interval_from_df(df, filename=f.filename)
            print(f"[run_regression_predictor] {f.filename} interval detected: {interval}")
            if interval == '5m' and file_5m is None:
                file_5m = (f, df)
        except Exception as e:
            print(f"[run_regression_predictor] Failed to read {f.filename}: {e}")
    if not file_5m:
        return jsonify({'feedback': 'A 5-minute CSV is required for regression simulation. Please upload a 5-minute interval CSV.'})
    f_5m, df_5m = file_5m
    df_5m.columns = [col.lower() for col in df_5m.columns]
    print(f"[run_regression_predictor] Using 5m file: {f_5m.filename}, shape: {df_5m.shape}")
    # Optionally, get user_params from form
    user_params = {}
    for k in ['max_risk_per_trade', 'max_daily_risk', 'long_threshold', 'short_threshold']:
        v = request.form.get(k)
        if v is not None:
            user_params[k] = v
    print(f"[run_regression_predictor] user_params: {user_params}")
    agent = RegressionPredictorAgent(user_params=user_params)
    fit_result = agent.fit(df_5m)
    print(f"[run_regression_predictor] agent.fit() returned: {fit_result}")
    # --- Progress tracker setup ---
    regression_backtest_tracker.update({
        "current": 0,
        "total": 0,
        "start_time": time.time(),
        "status": "running",
        "cancel_requested": False
    })
    result = agent.find_best_threshold_strategy(df_5m, user_params=user_params)
    regression_backtest_tracker["status"] = "done"
    print(f"[run_regression_predictor] agent.find_best_threshold_strategy() returned keys: {list(result.keys())}")
    print(f"[run_regression_predictor] Returning result to client.")

    result = to_serializable(result)
    # Pass through regression_trades_plot_path if present in best_result
    # Define which keys to keep
    keys_to_keep = [
        "llm_summary",
        "recommended_strategy_summary",
        "llm_top_strategies",
        "llm_model_name",
        "llm_token_usage",
        "llm_session_total_cost",
        "llm_cost_usd",
        "total_cost_usd",
    ]

    # Optional: include debug items like 'filter_meta' or 'strategy_matrix_llm' as needed
    if "filter_meta" in result:
        keys_to_keep.append("filter_meta")
    if "strategy_matrix_llm" in result:
        keys_to_keep.append("strategy_matrix_llm")

    # Build filtered dictionary
    cleaned_result = {k: result[k] for k in keys_to_keep if k in result}

    # Optional: add regression_trades_plot_path if relevant
    if 'regression_trades_plot_path' in result:
        cleaned_result['regression_trades_plot_path'] = result['regression_trades_plot_path']

    # Final conversion and return
    cleaned_result = to_serializable(cleaned_result)
    cleaned_result["llm_model_name"] = CONFIG.get('model_name')
    return jsonify(cleaned_result)


@app.route("/regression_backtest_progress", methods=["GET"])
def regression_backtest_progress():
    return jsonify(regression_backtest_tracker)

@app.route("/cancel_regression_backtest", methods=["POST"])
def cancel_regression_backtest():
    regression_backtest_tracker["cancel_requested"] = True
    return jsonify({"message": "Cancellation requested"})

@app.route("/run_vwap_agent", methods=["POST"])
def run_vwap_agent():
    """
    Accepts up to 4 images (field name: 'images'), determines the correct VWAP prompt, and calls the LLM (OpenAI or Gemini) with the prompt and images.
    Prints and returns the raw LLM response as JSON.
    """
    from flask import jsonify, request
    import traceback
    import json
    import pandas as pd
    from research_agent.scalp_agent.vwap_agent import VWAPAgent

    try:
        # Step 1: Get images from request
        images = request.files.getlist("images")
        if not images or len(images) == 0:
            return jsonify({"error": "No images uploaded. Please upload at least one image."}),
        if len(images) > 4:
            return jsonify({"error": "Maximum 4 images allowed."}),

        # Step 2: Convert images to bytes
        image_bytes_list = []
        for img in images:
            img_bytes = img.read()
            if not img_bytes:
                continue
            image_bytes_list.append(img_bytes)
        num_images = len(image_bytes_list)
        if num_images == 0:
            return jsonify({"error": "No valid images found."}),

        csv_file = request.files.get('csv_file')
        if csv_file and csv_file.filename:
            df_ohlcv = pd.read_csv(csv_file)
        else:
            df_ohlcv = None
        vwap_agent = VWAPAgent()
        result = vwap_agent.run(images=image_bytes_list, df_5m=df_ohlcv)

        return jsonify(result)

    except Exception as e:
        print(f"[VWAP_AGENT] Error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/regression_strategy_defaults", methods=["GET"])
def regression_strategy_defaults():
    """Return the regression strategy default parameters as JSON for frontend UI."""
    return jsonify(REGRESSION_STRATEGY_DEFAULTS)

def to_serializable(val):
    if isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [to_serializable(v) for v in val]
    elif isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    elif isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    else:
        return val

@app.route('/heatmap')
def serve_heatmap():
    # Serve from the absolute path to uploaded_csvs
    img_dir = os.path.join(os.path.dirname(__file__), 'uploaded_csvs')
    filename = 'heatmap_debug.png'
    # Debug print to verify path
    print(f"[DEBUG] Serving heatmap from: {img_dir}, file: {filename}")
    return send_from_directory(img_dir, filename)

@app.route('/uploaded_csvs/<path:filename>')
def serve_uploaded_csvs(filename):
    import os
    from flask import send_from_directory
    dir_path = os.path.join(os.path.dirname(__file__), 'uploaded_csvs')
    return send_from_directory(dir_path, filename)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)