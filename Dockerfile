# Use slim Python 3.10 image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (needed by Pillow, matplotlib, Tkinter, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    python3-tk \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"

# Copy and install dependencies
COPY requirements.txt .

# Install all packages except numpy and pandas_ta (due to compatibility issues)
RUN grep -v "^#\|pandas_ta\|numpy" requirements.txt | grep -v "^$" > cleaned_requirements.txt && \
    pip install --no-cache-dir -r cleaned_requirements.txt

# Install numpy and pandas_ta in the correct order


# ✅ PATCH for squeeze_pro bug (from numpy import NaN → nan)
RUN sed -i 's/from numpy import NaN as npNaN/from numpy import nan as npNaN/' $(find / -type f -name squeeze_pro.py 2>/dev/null | head -n 1)

# Optional: Verify critical packages
RUN python -c "import numpy; log.info('Numpy version:', numpy.__version__)"
RUN python -c "import pandas_ta; log.info('Pandas_ta version:', pandas_ta.__version__)"
RUN python -c "import requests; log.info('Requests installed successfully')"

# Copy source code (including app and research_agent folders)
COPY research_agent/ research_agent/
COPY backend/ backend/
COPY frontend/ frontend/
COPY requirements.txt .



# Ensure folders exist
RUN mkdir -p uploaded_csvs temp_uploads data && chmod 777 uploaded_csvs temp_uploads data

# Expose port expected by Cloud Run
EXPOSE 8080

# Entrypoint for Flask app (inside research_agent)
CMD ["python", "research_agent/app.py"]
