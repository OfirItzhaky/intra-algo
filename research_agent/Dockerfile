FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# First install pandas_ta (without other dependencies that would update numpy)
RUN pip install pandas_ta==0.3.14b0

# Then install other packages but skip pandas_ta
RUN pip install --no-cache-dir $(grep -v pandas_ta requirements.txt)

# Make sure numpy is exactly version 1.24.4
RUN pip install numpy==1.24.4 --force-reinstall

# Test that pandas_ta's squeeze_pro can be imported
RUN python -c "from pandas_ta.momentum.squeeze_pro import squeeze_pro; print('Successfully imported squeeze_pro')"

# Expose the port
EXPOSE 8080

COPY . .

# Run the app
CMD ["python", "app.py"]
