FROM python:3.12-slim

WORKDIR /app

# OS deps (keep light; add libgomp1 for lightgbm, g++ for some builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Faster pip + no .pyc
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Copy requirements and install EVERYTHING (no filtering)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# (Optional) squeeze_pro patch — make conditional to avoid 'no input files'
# RUN f="$(find /usr/local -type f -name squeeze_pro.py 2>/dev/null | head -n 1)" \
#  && if [ -n "$f" ]; then \
#       sed -i 's/from numpy import NaN as npNaN/from numpy import nan as npNaN/' "$f"; \
#     fi

# (Optional) sanity prints (not log.info)
RUN python - << 'PY'
import numpy, sys
print("NumPy:", numpy.__version__, file=sys.stderr)
try:
    import pandas_ta
    print("pandas_ta:", pandas_ta.__version__, file=sys.stderr)
except Exception as e:
    print("pandas_ta import failed:", e, file=sys.stderr)
PY

# Copy project
COPY . /app

# Ensure data dirs exist
RUN mkdir -p uploaded_csvs temp_uploads data && chmod 777 uploaded_csvs temp_uploads data

# Cloud Run port
ENV PORT=8080
EXPOSE 8080

# Start app (Flask example). Make sure your app reads PORT or binds to 0.0.0.0:8080
CMD ["python", "research_agent/app.py"]
