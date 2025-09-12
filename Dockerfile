FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git tini \
 && rm -rf /var/lib/apt/lists/*

# Leverage cache: install Python deps first
COPY requirements*.txt /tmp/
RUN python -m pip install --upgrade pip setuptools wheel && \
    if [ -f /tmp/requirements.txt ]; then pip install -r /tmp/requirements.txt; fi && \
    if [ -f /tmp/requirements-dev.txt ]; then pip install -r /tmp/requirements-dev.txt; fi

# Copy app
COPY . /app

# Drop root
RUN useradd -m -u 1000 -U appuser && chown -R appuser:appuser /app
# Use numeric UID:GID to satisfy strict runAsNonRoot checks
USER 1000:1000

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["python","-c","print('AgentForge image ready')"]