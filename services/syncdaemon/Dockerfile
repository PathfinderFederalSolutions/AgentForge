# syntax=docker/dockerfile:1
# Pinned base image (distroless for minimal attack surface)
FROM python:3.11-slim@sha256:6d2f6ec47bff9bf3eab1a47d6f75c8f2a0d11d3b43e31e7c3c6c5b4f3fb2c5cb

# Security: create non-root user and group
RUN useradd -u 10001 -r -s /usr/sbin/nologin appuser \
    && mkdir -p /app/services/syncdaemon/logs /agentforge/replay \
    && chown -R appuser:appuser /app /agentforge \
    && chmod -R 755 /app /agentforge

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn prometheus-client nats-py

COPY services/syncdaemon/app /app/services/syncdaemon/app

USER appuser
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "services.syncdaemon.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
