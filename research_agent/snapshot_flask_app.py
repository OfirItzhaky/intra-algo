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
  <div contenteditable="true" id="paste-area">Click here and paste image (Ctrl+V)</div>
  <input type="hidden" name="pasted_image" id="pasted_image_data">
  <br>
  <button type="submit">📸 Run Snapshot Analysis</button>
</form>

<form method="POST" action="/reset" class="form-section">
  <button type="submit">🔁 Reset</button>
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
document.getElementById('paste-area').addEventListener('paste', function(event) {
  const items = (event.clipboardData || event.originalEvent.clipboardData).items;
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (item.type.indexOf('image') === 0) {
      const file = item.getAsFile();
      const reader = new FileReader();
      reader.onload = function(evt) {
        document.getElementById("pasted_image_data").value = evt.target.result;
        document.getElementById("paste-area").innerHTML = "<b>✅ Image captured and ready to submit</b>";
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

    market_bias = getattr(analyzer, "market_bias", "Unknown")
    sectors = ", ".join(CONFIG.get("focus_sectors", []))
    events = CONFIG.get("summarized_events", [])[:3]
    top_events = "\n".join(events)

    daily_results = [{
        "filename": "🧭 Daily Market Highlights",
        "text": f"**Bias:** {market_bias}\n**Focus Sectors:** {sectors}\n**Key Events:**\n{top_events}"
    }, {
        "filename": "📰 Full Daily Summary",
        "text": f"{CONFIG['summarized_news']}\n\n📅 Events:\n{CONFIG['summarized_events']}"
    }, {
        "filename": "💰 Cost Summary",
        "text": f"Total Estimated LLM Cost: ${cost:.6f}"
    }]

    return redirect(url_for("index"))

@app.route("/image_analysis", methods=["POST"])
def image_analysis():
    global image_results
    base64_data = request.form.get("pasted_image")
    if not base64_data or not base64_data.startswith("data:image"):
        image_results = [{
            "filename": "❌ No image found",
            "text": "Please paste a valid image using Ctrl+V in the box above."
        }]
        return redirect(url_for("index"))

    try:
        base64_str = base64_data.split(";base64,")[1]
        image_bytes = base64.b64decode(base64_str)
        result = image_ai.analyze_image_with_bytes(image_bytes, rule_id="trend_strength")
        text = result.get("raw_output", "⚠️ No response received.")
    except Exception as e:
        text = f"❌ Error during image decoding or analysis: {e}"

    image_results = [{
        "filename": "🖼 Pasted Snapshot",
        "text": text
    }, {
        "filename": "💰 Cost Summary",
        "text": f"Estimated LLM Cost: ${fetchers.cost_usd:.6f}"
    }]

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
    print("🔐 Loaded OpenAI key (full):", repr(CONFIG["openai_api_key"]))
    print("Length of loaded key:", len(CONFIG["openai_api_key"]))
    app.run(debug=True)
