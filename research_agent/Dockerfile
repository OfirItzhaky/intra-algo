FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# First install all packages (some may pull newer numpy)
RUN pip install --no-cache-dir -r requirements.txt

# Then downgrade numpy explicitly AFTER everything
RUN pip install numpy==1.24.4 --force-reinstall

# Optional: print to confirm version
RUN python -c "import numpy; print('✅ Final Numpy version:', numpy.__version__)"

COPY . .

CMD ["python", "app.py"]
