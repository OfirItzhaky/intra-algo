from flask import Flask, request, render_template_string, redirect, url_for
from datetime import datetime
import base64

from research_fetchers import ResearchFetchers, summarize_with_cache, summarize_economic_events_with_cache
from research_analyzer import ResearchAnalyzer
from image_analyzer_ai import ImageAnalyzerAI
from news_aggregator import NewsAggregator

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

# === Instantiate components ===
fetchers = ResearchFetchers(config=CONFIG)
aggregator = NewsAggregator(config=CONFIG)
analyzer = ResearchAnalyzer(config=CONFIG, fetchers=fetchers, aggregator=aggregator)

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

# === HTML UI ===
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
    h1, h2 { color: #111; }
    h1 { display: flex; align-items: center; }
    h1::before {
        content: "📸";
        font-size: 1.5rem;
        margin-right: 10px;
    }
    .instructions {
        background: #f8f8f8;
        border-left: 5px solid #ccc;
        padding: 10px 20px;
        margin-bottom: 20px;
        line-height: 1.6;
    }
    .form-section {
        margin-bottom: 30px;
    }
    #paste-area {
        width: 400px;
        height: 200px;
        border: 2px dashed #aaa;
        padding: 10px;
        font-style: italic;
        background: #fcfcfc;
    }
    #preview-area {
      display: flex;
      flex-wrap: wrap;
      margin-top: 10px;
      gap: 8px;
    }
    #preview-area img {
      width: 80px;
      height: auto;
      border: 1px solid #ccc;
      box-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    button {
        margin-top: 10px;
        padding: 6px 14px;
        font-weight: bold;
        background: #e6f0ff;
        border: 1px solid #999;
        cursor: pointer;
    }
    pre {
        background: #f4f4f4;
        padding: 10px;
        border-radius: 5px;
        max-width: 100%;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .results-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
    }
    .result-box {
        width: 48%;
        word-wrap: break-word;
    }
  </style>
</head>
<body>

<h1>Snapshot Trade Analyzer (Flask UI)</h1>

<div class="instructions">
  <p><b>📊 Overview:</b> This tool performs two types of analysis when you click one of the buttons below:</p>
  <ul>
    <li><b>Market Bias Generator</b> — analyzes today's <u>news and economic events</u> to infer general market tone (Bullish / Neutral / Bearish).</li>
    <li><b>Snapshot Image Analyzer</b> — allows you to paste <u>technical indicator charts</u> like $NYAD, $NYHL etc. for rule-based evaluation.</li>
  </ul>
</div>

<form method="POST" action="/daily_analysis" class="form-section">
  <button type="submit">🧠 Run Daily Market Summary</button>
</form>

<form method="POST" enctype="multipart/form-data" action="/image_analysis" class="form-section">
  <label><b>Paste your image here:</b></label><br>
  <div contenteditable="true" id="paste-area">
      Click here and paste image (Ctrl+V)
      <div id="preview-area"></div>
    </div>
  <input type="hidden" id="image_count" name="image_count" value="0">
  <br>
  <button type="submit">📸 Run Snapshot Analysis</button>
</form>

<form method="POST" action="/reset" class="form-section">
  <button type="submit">🔁 Reset</button>
</form>
<form method="POST" action="/momentum_analysis" class="form-section" target="_blank">
  <button type="submit">📈 Generate Momentum Report</button>
</form>



<div class="results-container">
  <div class="result-box">
    {% if daily_outputs %}
      <h2>📊 Daily Market Outputs</h2>
      {% for result in daily_outputs %}
        <h4>{{ result.filename }}</h4>
        <pre>{{ result.text }}</pre>
      {% endfor %}
    {% endif %}
  </div>
  <div class="result-box">
    {% if image_outputs %}
      <h2>📸 Snapshot Outputs</h2>
      {% for result in image_outputs %}
        <h4>{{ result.filename }}</h4>
        <pre>{{ result.text }}</pre>
      {% endfor %}
    {% endif %}
  </div>
</div>

<script>
const pasteArea = document.getElementById('paste-area');
const previewArea = document.getElementById("preview-area");
const imageCountInput = document.getElementById("image_count");
let imageIndex = 1;

pasteArea.addEventListener('paste', function(event) {
  event.preventDefault();  // prevent default paste
  const items = (event.clipboardData || event.originalEvent.clipboardData).items;

  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (item.type.indexOf('image') === 0) {
      const file = item.getAsFile();
      const reader = new FileReader();
      reader.onload = function(evt) {
        const img = document.createElement("img");
        img.src = evt.target.result;
        img.style.width = "80px";
        img.style.margin = "4px";
        previewArea.appendChild(img);

        // ⬇️ Create hidden input for each image
        const input = document.createElement("input");
        input.type = "hidden";
        input.name = `image_${imageIndex}`;
        input.value = evt.target.result;
        pasteArea.parentNode.appendChild(input);

        imageIndex += 1;
        imageCountInput.value = imageIndex - 1;
      };
      reader.readAsDataURL(file);
    }
  }
});
</script>



</body>
</html>
'''


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, daily_outputs=daily_results, image_outputs=image_results)

@app.route("/daily_analysis", methods=["POST"])
def daily_analysis():
    global daily_results
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

    return redirect(url_for("index"))

@app.route("/momentum_analysis", methods=["POST"])
def momentum_analysis():
    from market_momentum_scorer import MarketMomentumScorer

    scorer = MarketMomentumScorer()
    scorer.fetch_data()
    scorer.compute_indicators()

    results_df = scorer.build_summary_table()

    # Create the HTML table using Plotly
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[
                "Symbol",
                "Weekly Momentum", "Touched MA (Weekly)",
                "Daily Momentum", "Touched MA (Daily)"
            ],
            fill_color="lightgray",
            align="center",
            font=dict(size=14)
        ),
        cells=dict(
            values=[
                results_df["symbol"],
                results_df["momentum_color_weekly"],
                results_df["touch_recent_ma_weekly"].map({True: "Yes", False: "No"}),
                results_df["momentum_color_daily"],
                results_df["touch_recent_ma_daily"].map({True: "Yes", False: "No"})
            ],
            fill_color="white",
            align="center",
            font=dict(size=12),
            height=28
        )
    )])
    fig.update_layout(title_text="Momentum Table (Weekly + Daily)", margin=dict(t=50, b=20))

    html_page = f"""
    <html>
    <head><title>Momentum Tables</title></head>
    <body style="font-family: Arial; margin: 40px;">
      <h1>📊 Momentum Summary</h1>
      {fig.to_html(full_html=False)}
    </body>
    </html>
    """
    return html_page






@app.route("/image_analysis", methods=["POST"])
def image_analysis():
    global image_results
    image_results = []

    image_count = int(request.form.get("image_count", 0))

    if image_count == 0:
        image_results = [{
            "filename": "❌ No images pasted",
            "text": "Please paste one or more images using Ctrl+V in the box above."
        }]
        return redirect(url_for("index"))

    for i in range(1, image_count + 1):
        base64_data = request.form.get(f"image_{i}")
        if not base64_data or not base64_data.startswith("data:image"):
            image_results.append({
                "filename": f"❌ Image {i} invalid",
                "text": "This pasted image is invalid or missing."
            })
            continue

        try:
            base64_str = base64_data.split(";base64,")[1]
            image_bytes = base64.b64decode(base64_str)
            result = image_ai.analyze_image_with_bytes(image_bytes, rule_id="trend_strength")
            text = result.get("raw_output", "⚠️ No response received.")
        except Exception as e:
            text = f"❌ Error analyzing image {i}: {e}"

        image_results.append({
            "filename": f"🖼 Snapshot {i}",
            "text": text
        })

    image_results.append({
        "filename": "💰 Cost Summary",
        "text": f"Estimated LLM Cost: ${fetchers.cost_usd:.6f}"
    })

    return redirect(url_for("index"))


@app.route("/reset", methods=["POST"])
def reset():
    global daily_results, image_results
    SUMMARY_CACHE.clear()
    EVENT_CACHE.clear()
    fetchers.token_usage = 0
    fetchers.cost_usd = 0
    daily_results = []
    image_results = []
    return redirect(url_for("index"))

if __name__ == "__main__":
    print("Length of loaded key:", len(CONFIG["openai_api_key"]))
    app.run(debug=True, use_reloader=False)

