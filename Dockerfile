# MoleculeIQ - Production Docker Container
# AI-Assisted Molecular Screening and Cheminformatics Platform

FROM python:3.12-slim

# Prevent interactive prompts and Python output buffering
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

# Install system dependencies for scientific libraries and RDKit rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libxrender1 \
    libxext6 \
    libsm6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root application user for container security
RUN groupadd -r moleculeiq && useradd -r -g moleculeiq -d /app -s /sbin/nologin moleculeiq

WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code, models, reference data, and configuration
COPY src/ src/
COPY artifacts/ artifacts/
COPY data/reference/ data/reference/
COPY app.py .

# Ensure ownership by non-root user
RUN chown -R moleculeiq:moleculeiq /app

USER moleculeiq

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
