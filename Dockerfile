FROM python:3.10-slim

# Install system dependencies (needed by some ML libs)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ✅ Copy requirements FIRST (Docker layer cache)
# Only reinstalls if requirements.txt changes
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ✅ Copy app code AFTER dependencies
COPY . .

# ✅ Pre-compile Python files (faster startup)
RUN python -m compileall -q .

CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "3600", \
     "--graceful-timeout", "3600", \
     "--keep-alive", "300", \
     "--workers", "1", \
     "--threads", "8", \
     "--worker-class", "gthread", \
     "--preload", \
     "app:app"]