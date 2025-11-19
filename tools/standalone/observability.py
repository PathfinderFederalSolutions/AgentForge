# observability.py
import logging
from logging import Formatter
import json
import os
from prometheus_client import Counter, Histogram, start_http_server

class JsonFormatter(Formatter):
    def format(self, record):
        data = {"level": record.levelname, "msg": record.msg}
        if hasattr(record, 'task_id'):
            data["task_id"] = record.task_id
        if hasattr(record, 'request_id'):
            data["request_id"] = record.request_id
        return json.dumps(data)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# Safely set formatter if a handler exists
if logger.handlers:
    logger.handlers[0].setFormatter(JsonFormatter())

# Metrics
token_counter = Counter('tokens_used', 'Tokens used', ['provider', 'agent'])
latency_hist = Histogram('request_latency_seconds', 'Latency', ['provider', 'agent'])
cost_counter = Counter('cost', 'Cost', ['provider', 'agent'])
error_counter = Counter('errors', 'Errors', ['provider', 'agent'])

def log_with_id(task_id: str, msg: str, request_id: str = None):
    extra = {'task_id': task_id}
    if request_id:
        extra['request_id'] = request_id
    logger_adapter = logging.LoggerAdapter(logger, extra)
    logger_adapter.info(msg)

# Start Prometheus server only when enabled
_prometheus_enabled = os.getenv("PROMETHEUS_ENABLE", "0") == "1"
_prometheus_port = int(os.getenv("PROMETHEUS_PORT", "8000"))
if _prometheus_enabled:
    try:
        start_http_server(_prometheus_port)  # Access at http://localhost:PORT/metrics
    except OSError:
        # Port may already be in use in test runs; ignore
        pass