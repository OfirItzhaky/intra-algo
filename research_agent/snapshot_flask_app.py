from flask import Flask, request, render_template_string, redirect, url_for
from werkzeug.utils import secure_filename
from datetime import datetime

from intra_algo_research_agent_for_flask import (
    ResearchAnalyzer, ResearchFetchers, CONFIG,
    SUMMARY_CACHE, EVENT_CACHE, ImageAnalyzerAI,
    summarize_with_cache, summarize_economic_events_with_cache
)

app = Flask(__name__)
app.secret_key = 'snapshot-session-key'

# === Instantiate components ===
fetchers = ResearchFetchers(config=CONFIG)
analyzer = ResearchAnalyzer(config=CONFIG, fetchers=fetchers)
image_ai = ImageAnalyzerAI(
    model_provider=CONFIG["model_provider"],
    model_name=CONFIG["model_name"],
    api_key=CONFIG["openai_api_key"] if CONFIG["model_provider"] == "openai" else CONFIG["gemini_api_key"]
)

# === Prepare context without running anything on import ===
def prepare_daily_context():
    merged_headlines = fetchers.aggregate_news()
    summarized_news = summarize_with_cache(fetchers, merged_headlines, force_refresh=False)
    summarized_events = summarize_economic_events_with_cache(fetchers, force_refresh=False)

    CONFIG["summarized_news"] = summarized_news
    CONFIG["summarized_events"] = summarized_events

# === HTML UI ===
HTML_TEMPLATE = '''
<h1>📸 Snapshot Trade Analyzer (Flask UI)</h1>

<form method="POST" enctype="multipart/form-data" action="/analyze">
  <label><b>Upload snapshot image(s):</b></label><br>
  <input type="file" name="images" accept="image/*" multiple required><br><br>
  <button type="submit">▶️ Run Analysis</button>
</form>

<br>

<form method="POST" action="/reset">
  <button type="submit">🔁 Reset</button>
</form>

{% if outputs %}
<hr>
<h2>📊 Raw LLM Outputs</h2>
{% for result in outputs %}
  <h4>{{ result.filename }}</h4>
  <pre style="background:#f4f4f4;padding:10px;">{{ result.text }}</pre>
{% endfor %}
{% endif %}
'''

# === Home Page ===
@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, outputs=None)

# === Main Analysis Route ===
@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded_files = request.files.getlist("images")
    results = []

    # Step 1: Run summaries & analyzer (safe to call here)
    prepare_daily_context()
    analyzer.run_daily_analysis()
    summary_cost = fetchers.cost_usd

    # Step 2: Analyze each uploaded image
    for file in uploaded_files:
        image_bytes = file.read()
        filename = secure_filename(file.filename)

        try:
            result = image_ai.analyze_image_with_bytes(image_bytes, rule_id="trend_strength")
            text = result.get("raw_output", "⚠️ No response received.")
        except Exception as e:
            text = f"❌ Error analyzing {filename}: {e}"

        results.append({
            "filename": filename,
            "text": text
        })

    # Step 3: Print total LLM cost (summary + images)
    total_cost = summary_cost + fetchers.cost_usd
    results.append({
        "filename": "💰 Cost Summary",
        "text": f"Total Estimated LLM Cost: ${total_cost:.4f}"
    })

    return render_template_string(HTML_TEMPLATE, outputs=results)

# === Reset Cache ===
@app.route("/reset", methods=["POST"])
def reset():
    SUMMARY_CACHE.clear()
    EVENT_CACHE.clear()
    fetchers.token_usage = 0
    fetchers.cost_usd = 0
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
