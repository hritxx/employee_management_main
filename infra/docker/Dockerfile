# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=5

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    postgresql-client \
    gzip \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY ../../requirements.txt .

# Install Python dependencies with retry and timeout handling
# Install base dependencies first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core dependencies
RUN pip install --no-cache-dir --retries 5 --timeout 300 \
    streamlit==1.29.0 \
    sqlalchemy==1.4.53 \
    psycopg2-binary==2.9.9 \
    python-dotenv==1.0.0 \
    pyyaml==6.0.1 \
    requests==2.31.0 \
    python-dateutil==2.8.2 \
    bcrypt==4.1.2

# Install remaining dependencies one by one for better error handling
RUN pip install --no-cache-dir --retries 5 --timeout 300 numpy==1.24.4 || \
    pip install --no-cache-dir --retries 3 --timeout 180 numpy==1.24.3 || \
    pip install --no-cache-dir --retries 3 --timeout 120 numpy

RUN pip install --no-cache-dir --retries 5 --timeout 300 pandas==1.5.3 || \
    pip install --no-cache-dir --retries 3 --timeout 180 pandas

RUN pip install --no-cache-dir --retries 5 --timeout 300 \
    google-generativeai==0.3.2 \
    openpyxl==3.1.2 \
    plotly==5.17.0 \
    reportlab==4.0.8

# Copy the entire application
COPY ../../ .

# Create necessary directories
RUN mkdir -p logs data uploads data/backups

# Set permissions
RUN chmod +x /app

# Expose the port that Streamlit runs on
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]