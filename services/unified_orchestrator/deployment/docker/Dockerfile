# Multi-stage production Dockerfile for Unified Quantum Orchestrator
# Optimized for security, performance, and minimal attack surface

# Stage 1: Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for build
RUN useradd --create-home --shell /bin/bash agi

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements*.txt ./
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=agi:agi . .

# Install the application
RUN pip install --no-cache-dir -e .

# Stage 2: Production runtime
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 agi

# Create necessary directories
RUN mkdir -p /app /var/log/agi /var/lib/agi /etc/agi-orchestrator && \
    chown -R agi:agi /app /var/log/agi /var/lib/agi /etc/agi-orchestrator

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder --chown=agi:agi /app /app

# Set working directory
WORKDIR /app

# Switch to non-root user
USER agi

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

# Expose ports
EXPOSE 8080 9090

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO \
    ENVIRONMENT=production

# Default command
CMD ["python", "-m", "services.unified_orchestrator.main"]
