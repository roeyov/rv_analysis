FROM python:3.11-slim

# System dependencies for matplotlib (headless) and scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Headless matplotlib backend
ENV MPLBACKEND=Agg

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline code
COPY . .

# Default: run the cloud worker
ENTRYPOINT ["python", "-m", "cloud.worker"]
