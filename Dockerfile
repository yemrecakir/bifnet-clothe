FROM python:3.10-slim

# Install system dependencies with build optimization
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    wget \
    libffi-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy application code
COPY . .

# Create temp directory for API processing
RUN mkdir -p temp_api

# Pre-download BiRefNet model during build for faster cold starts
RUN python download_model.py && echo "✅ Model pre-downloaded successfully" || echo "⚠️ Model download failed, will download on first request"

# Set environment variables for performance
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV TORCH_HOME=/app/models
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4

# Expose port (Cloud Run uses PORT env var)
EXPOSE 8080

# Health check with optimized timing
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT:-8080}/health', timeout=10)" || exit 1

# Start the optimized API server (use PORT env var for Cloud Run compatibility)
CMD ["sh", "-c", "uvicorn optimized_api_server:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --loop uvloop --http httptools"]