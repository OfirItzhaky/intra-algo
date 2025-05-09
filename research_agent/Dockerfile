# Use Python 3.12 instead of 3.10
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install pip and system dependencies
RUN apt-get update && apt-get install -y gcc && apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Start the app
CMD ["python", "app.py"]
# Add this line for debug
RUN python -c "import numpy; print('✅ Numpy version:', numpy.__version__)"