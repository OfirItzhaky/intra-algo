# Use slim Python 3.10 image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (needed by Pillow, Tkinter, matplotlib, etc.)
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

# Split install for numpy/pandas_ta compatibility
RUN grep -v "^#\|pandas_ta\|numpy" requirements.txt | grep -v "^$" > cleaned_requirements.txt && \
    pip install --no-cache-dir -r cleaned_requirements.txt

# Install numpy and pandas_ta separately to control versioning
RUN pip install numpy==1.24.4
RUN pip install pandas_ta==0.3.14b0

# Optional: Check core package versions
RUN python -c "import numpy; print('✅ Numpy version:', numpy.__version__)"
RUN python -c "import pandas_ta; print('✅ Pandas_ta version:', pandas_ta.__version__)"
RUN python -c "import requests; print('✅ Requests works!')"

# Copy full source code
COPY . .

# Ensure critical folders exist
RUN mkdir -p uploaded_csvs temp_uploads data && chmod 777 uploaded_csvs temp_uploads data

# Verify pandas_ta patch if applicable
RUN python -c "import fix_numpy; from pandas_ta.momentum.squeeze_pro import squeeze_pro; print('✅ squeeze_pro patch success')"

# Expose port expected by Cloud Run
EXPOSE 8080

# Entrypoint
CMD ["python", "app.py"]
