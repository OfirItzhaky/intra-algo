FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install Flask and other basic packages except pandas_ta and numpy
RUN pip install --no-cache-dir $(grep -v "pandas_ta\|numpy" requirements.txt)

# Install numpy first with exact version
RUN pip install numpy==1.24.4

# Then install pandas_ta 
RUN pip install pandas_ta==0.3.14b0

# Verify packages
RUN python -c "import numpy; print('Numpy version:', numpy.__version__)"
RUN python -c "import pandas_ta; print('Pandas_ta version:', pandas_ta.__version__)"

# Copy all code first
COPY . .

# Test that our fix works
RUN python -c "import fix_numpy; from pandas_ta.momentum.squeeze_pro import squeeze_pro; print('Successfully imported squeeze_pro with our fix')"

# Expose the port
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]
