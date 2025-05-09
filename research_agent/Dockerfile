FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for Pillow and Tkinter
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    python3-tk \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable to indicate production environment
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

# Install all packages except pandas_ta and numpy
RUN grep -v "^#\|pandas_ta\|numpy" requirements.txt | grep -v "^$" > cleaned_requirements.txt && \
    pip install --no-cache-dir -r cleaned_requirements.txt

# Install numpy with specific version
RUN pip install numpy==1.24.4

# Then install pandas_ta
RUN pip install pandas_ta==0.3.14b0

# Verify core packages
RUN python -c "import numpy; print('Numpy version:', numpy.__version__)"
RUN python -c "import pandas_ta; print('Pandas_ta version:', pandas_ta.__version__)"
RUN python -c "import requests; print('Requests installed successfully')"

# Copy all code
COPY . .

# Test our fix works
RUN python -c "import fix_numpy; from pandas_ta.momentum.squeeze_pro import squeeze_pro; print('Successfully imported squeeze_pro with our fix')"

# Expose the port that Flask will run on
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]
