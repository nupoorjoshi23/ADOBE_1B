# Use a slim, modern Python version as the base
FROM python:3.11-slim

WORKDIR /app

# --- THE FIX IS HERE ---
# Install system dependencies with the correct package name
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
# ----------------------

# Copy requirements and install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Pre-downloading ALL Models and NLP data for Offline Use ---
COPY src/download_models.py /app/src/download_models.py
RUN python -m src.download_models

# Copy your application code and pre-downloaded models
COPY src/ /app/src/
COPY models/ /app/models/

# The command that will be run when the container starts
CMD ["python", "-m", "src.main"]