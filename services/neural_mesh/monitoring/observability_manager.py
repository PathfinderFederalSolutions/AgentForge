"""
Observability Manager - Comprehensive Telemetry and Monitoring for Neural Mesh
Implements distributed tracing, metrics collection, rate limiting, and predictive alerting
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum
from collections import defaultdict, deque
import statistics

# Optional imports
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

try:
    import opentelemetry
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    opentelemetry = None
    trace = None
    metrics = None
    OPENTELEMETRY_AVAILABLE = False

# Metrics imports
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    Counter = Histogram = Gauge = Summary = lambda *args, **kwargs: None

log = logging.getLogger("observability-manager")

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class TraceSpan:
    """Distributed tracing span"""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    service_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # ok, error, timeout
    
    def finish(self, status: str = "ok"):
        """Finish the span"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
    
    def add_tag(self, key: str, value: Any):
        """Add tag to span"""
        self.tags[key] = value
    
    def add_log(self, message: str, **fields):
        """Add log entry to span"""
        self.logs.append({
            "timestamp": time.time(),
            "message": message,
            **fields
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return asdict(self)

@dataclass
class Alert:
    """Alert definition"""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    message: str = ""
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None
    
    def resolve(self):
        """Mark alert as resolved"""
        self.resolved = True
        self.resolved_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return asdict(self)

class RateLimiter:
    """Advanced rate limiter with multiple strategies"""
    
    def __init__(self, strategy: RateLimitStrategy, limit: int, window_seconds: float = 60.0):
        self.strategy = strategy
        self.limit = limit
        self.window_seconds = window_seconds
        
        # Strategy-specific state
        if strategy == RateLimitStrategy.FIXED_WINDOW:
            self.window_start = time.time()
            self.current_count = 0
        elif strategy == RateLimitStrategy.SLIDING_WINDOW:
            self.requests = deque()
        elif strategy == RateLimitStrategy.TOKEN_BUCKET:
            self.tokens = limit
            self.last_refill = time.time()
            self.refill_rate = limit / window_seconds
        elif strategy == RateLimitStrategy.LEAKY_BUCKET:
            self.queue = deque()
            self.last_leak = time.time()
            self.leak_rate = limit / window_seconds
        
        self.lock = threading.Lock()
    
    def is_allowed(self, tokens: int = 1) -> bool:
        """Check if request is allowed"""
        with self.lock:
            current_time = time.time()
            
            if self.strategy == RateLimitStrategy.FIXED_WINDOW:
                return self._fixed_window_check(current_time, tokens)
            elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return self._sliding_window_check(current_time, tokens)
            elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return self._token_bucket_check(current_time, tokens)
            elif self.strategy == RateLimitStrategy.LEAKY_BUCKET:
                return self._leaky_bucket_check(current_time, tokens)
            else:
                return True  # Default allow
    
    def _fixed_window_check(self, current_time: float, tokens: int) -> bool:
        """Fixed window rate limiting"""
        # Reset window if expired
        if current_time - self.window_start >= self.window_seconds:
            self.window_start = current_time
            self.current_count = 0
        
        if self.current_count + tokens <= self.limit:
            self.current_count += tokens
            return True
        return False
    
    def _sliding_window_check(self, current_time: float, tokens: int) -> bool:
        """Sliding window rate limiting"""
        # Remove old requests
        while self.requests and current_time - self.requests[0] > self.window_seconds:
            self.requests.popleft()
        
        if len(self.requests) + tokens <= self.limit:
            for _ in range(tokens):
                self.requests.append(current_time)
            return True
        return False
    
    def _token_bucket_check(self, current_time: float, tokens: int) -> bool:
        """Token bucket rate limiting"""
        # Refill tokens
        elapsed = current_time - self.last_refill
        self.tokens = min(self.limit, self.tokens + elapsed * self.refill_rate)
        self.last_refill = current_time
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _leaky_bucket_check(self, current_time: float, tokens: int) -> bool:
        """Leaky bucket rate limiting"""
        # Leak requests
        elapsed = current_time - self.last_leak
        leak_count = int(elapsed * self.leak_rate)
        for _ in range(min(leak_count, len(self.queue))):
            self.queue.popleft()
        self.last_leak = current_time
        
        if len(self.queue) + tokens <= self.limit:
            for _ in range(tokens):
                self.queue.append(current_time)
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        with self.lock:
            current_time = time.time()
            
            if self.strategy == RateLimitStrategy.FIXED_WINDOW:
                remaining = max(0, self.limit - self.current_count)
                reset_time = self.window_start + self.window_seconds
            elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                # Clean old requests for accurate count
                while self.requests and current_time - self.requests[0] > self.window_seconds:
                    self.requests.popleft()
                remaining = max(0, self.limit - len(self.requests))
                reset_time = current_time + self.window_seconds
            elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                # Update tokens
                elapsed = current_time - self.last_refill
                tokens = min(self.limit, self.tokens + elapsed * self.refill_rate)
                remaining = int(tokens)
                reset_time = current_time + (self.limit - tokens) / self.refill_rate
            elif self.strategy == RateLimitStrategy.LEAKY_BUCKET:
                remaining = max(0, self.limit - len(self.queue))
                reset_time = current_time + len(self.queue) / self.leak_rate
            else:
                remaining = self.limit
                reset_time = current_time
            
            return {
                "strategy": self.strategy.value,
                "limit": self.limit,
                "remaining": remaining,
                "reset_time": reset_time,
                "window_seconds": self.window_seconds
            }

class MetricsCollector:
    """Comprehensive metrics collection system"""
    
    def __init__(self, service_name: str = "neural-mesh"):
        self.service_name = service_name
        
        # Metrics storage
        self.metrics: Dict[str, Any] = {}
        self.custom_metrics: Dict[str, Callable] = {}
        
        # Prometheus registry
        if METRICS_AVAILABLE:
            self.registry = CollectorRegistry()
            self._setup_default_metrics()
        
        # Time series data for analysis
        self.time_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Background collection
        self.collection_tasks = []
        self.is_collecting = False
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def _setup_default_metrics(self):
        """Setup default Prometheus metrics"""
        if not METRICS_AVAILABLE:
            return
        
        # System metrics
        self.system_memory_usage = Gauge(
            'neural_mesh_system_memory_bytes',
            'System memory usage',
            registry=self.registry
        )
        
        self.system_cpu_usage = Gauge(
            'neural_mesh_system_cpu_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        # Neural mesh specific metrics
        self.memory_operations_total = Counter(
            'neural_mesh_memory_operations_total',
            'Total memory operations',
            ['tier', 'operation', 'status'],
            registry=self.registry
        )
        
        self.memory_operation_duration = Histogram(
            'neural_mesh_memory_operation_duration_seconds',
            'Memory operation duration',
            ['tier', 'operation'],
            registry=self.registry
        )
        
        self.pattern_detection_operations = Counter(
            'neural_mesh_pattern_detection_total',
            'Pattern detection operations',
            ['pattern_type', 'status'],
            registry=self.registry
        )
        
        self.knowledge_synthesis_operations = Counter(
            'neural_mesh_knowledge_synthesis_total',
            'Knowledge synthesis operations',
            ['knowledge_type', 'status'],
            registry=self.registry
        )
        
        self.active_agents = Gauge(
            'neural_mesh_active_agents',
            'Number of active agents',
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'neural_mesh_cache_hit_rate',
            'Cache hit rate',
            ['cache_type'],
            registry=self.registry
        )
    
    async def start_collection(self):
        """Start background metrics collection"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.collection_tasks = [
            asyncio.create_task(self._system_metrics_collector()),
            asyncio.create_task(self._custom_metrics_collector())
        ]
        
        log.info("Started metrics collection")
    
    async def stop_collection(self):
        """Stop background metrics collection"""
        self.is_collecting = False
        
        for task in self.collection_tasks:
            task.cancel()
        
        await asyncio.gather(*self.collection_tasks, return_exceptions=True)
        self.collection_tasks.clear()
        
        log.info("Stopped metrics collection")
    
    async def _system_metrics_collector(self):
        """Collect system metrics"""
        while self.is_collecting:
            try:
                import psutil
                
                # Memory usage
                memory = psutil.virtual_memory()
                if METRICS_AVAILABLE:
                    self.system_memory_usage.set(memory.used)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                if METRICS_AVAILABLE:
                    self.system_cpu_usage.set(cpu_percent)
                
                # Store in time series
                current_time = time.time()
                with self.lock:
                    self.time_series["system.memory.used"].append((current_time, memory.used))
                    self.time_series["system.cpu.percent"].append((current_time, cpu_percent))
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                log.error(f"System metrics collection error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    
    async def _custom_metrics_collector(self):
        """Collect custom metrics"""
        while self.is_collecting:
            try:
                current_time = time.time()
                
                with self.lock:
                    for metric_name, collector_func in self.custom_metrics.items():
                        try:
                            value = collector_func()
                            self.time_series[metric_name].append((current_time, value))
                        except Exception as e:
                            log.debug(f"Custom metric {metric_name} collection failed: {e}")
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                log.error(f"Custom metrics collection error: {e}")
                await asyncio.sleep(30)
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                     labels: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        current_time = time.time()
        labels = labels or {}
        
        with self.lock:
            # Store in time series
            metric_key = f"{name}.{'.'.join(f'{k}={v}' for k, v in labels.items())}"
            self.time_series[metric_key].append((current_time, value))
            
            # Update Prometheus metrics
            if METRICS_AVAILABLE and hasattr(self, f"custom_{name}"):
                metric = getattr(self, f"custom_{name}")
                if metric_type == MetricType.COUNTER:
                    metric.labels(**labels).inc(value)
                elif metric_type == MetricType.GAUGE:
                    metric.labels(**labels).set(value)
                elif metric_type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
                    metric.labels(**labels).observe(value)
    
    def register_custom_metric(self, name: str, collector_func: Callable) -> bool:
        """Register custom metric collector"""
        try:
            with self.lock:
                self.custom_metrics[name] = collector_func
            log.info(f"Registered custom metric: {name}")
            return True
        except Exception as e:
            log.error(f"Failed to register custom metric {name}: {e}")
            return False
    
    def get_metric_stats(self, metric_name: str, time_window: float = 3600) -> Dict[str, Any]:
        """Get statistics for a metric over time window"""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self.lock:
            if metric_name not in self.time_series:
                return {"error": "Metric not found"}
            
            # Filter data points within time window
            data_points = [
                (timestamp, value) for timestamp, value in self.time_series[metric_name]
                if timestamp >= cutoff_time
            ]
            
            if not data_points:
                return {"error": "No data points in time window"}
            
            values = [value for _, value in data_points]
            
            return {
                "metric_name": metric_name,
                "time_window": time_window,
                "data_points": len(data_points),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                "latest_value": values[-1],
                "latest_timestamp": data_points[-1][0]
            }
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if METRICS_AVAILABLE:
            return generate_latest(self.registry).decode('utf-8')
        return ""

class DistributedTracer:
    """Distributed tracing system"""
    
    def __init__(self, service_name: str = "neural-mesh"):
        self.service_name = service_name
        
        # Active spans
        self.active_spans: Dict[str, TraceSpan] = {}
        self.span_stack: List[str] = []  # Thread-local span stack
        
        # Span storage
        self.completed_spans: deque = deque(maxlen=10000)
        
        # OpenTelemetry setup
        if OPENTELEMETRY_AVAILABLE:
            self._setup_opentelemetry()
        
        # Background export
        self.export_tasks = []
        self.is_exporting = False
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry tracing"""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "1.0.0"
            })
            
            # Setup tracer provider
            tracer_provider = TracerProvider(resource=resource)
            
            # Add Jaeger exporter if available
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            tracer_provider.add_span_processor(span_processor)
            
            # Set as global tracer provider
            trace.set_tracer_provider(tracer_provider)
            self.tracer = trace.get_tracer(__name__)
            
            log.info("OpenTelemetry tracing initialized")
            
        except Exception as e:
            log.warning(f"OpenTelemetry setup failed: {e}")
            self.tracer = None
    
    def start_span(self, operation_name: str, parent_span_id: Optional[str] = None,
                   tags: Optional[Dict[str, Any]] = None) -> TraceSpan:
        """Start a new trace span"""
        with self.lock:
            # Determine parent
            if parent_span_id is None and self.span_stack:
                parent_span_id = self.span_stack[-1]
            
            # Create span
            span = TraceSpan(
                operation_name=operation_name,
                service_name=self.service_name,
                parent_span_id=parent_span_id,
                tags=tags or {}
            )
            
            # Set trace ID from parent or create new one
            if parent_span_id and parent_span_id in self.active_spans:
                parent_span = self.active_spans[parent_span_id]
                span.trace_id = parent_span.trace_id
            
            # Store active span
            self.active_spans[span.span_id] = span
            self.span_stack.append(span.span_id)
            
            return span
    
    def finish_span(self, span: TraceSpan, status: str = "ok"):
        """Finish a trace span"""
        with self.lock:
            span.finish(status)
            
            # Remove from active spans
            if span.span_id in self.active_spans:
                del self.active_spans[span.span_id]
            
            # Remove from stack
            if self.span_stack and self.span_stack[-1] == span.span_id:
                self.span_stack.pop()
            
            # Store completed span
            self.completed_spans.append(span)
            
            log.debug(f"Finished span {span.operation_name} in {span.duration:.3f}s")
    
    def get_current_span(self) -> Optional[TraceSpan]:
        """Get current active span"""
        with self.lock:
            if self.span_stack:
                span_id = self.span_stack[-1]
                return self.active_spans.get(span_id)
            return None
    
    def add_span_tag(self, key: str, value: Any):
        """Add tag to current span"""
        current_span = self.get_current_span()
        if current_span:
            current_span.add_tag(key, value)
    
    def add_span_log(self, message: str, **fields):
        """Add log to current span"""
        current_span = self.get_current_span()
        if current_span:
            current_span.add_log(message, **fields)
    
    async def start_export(self):
        """Start background span export"""
        if self.is_exporting:
            return
        
        self.is_exporting = True
        self.export_tasks = [
            asyncio.create_task(self._span_exporter())
        ]
        
        log.info("Started distributed tracing export")
    
    async def stop_export(self):
        """Stop background span export"""
        self.is_exporting = False
        
        for task in self.export_tasks:
            task.cancel()
        
        await asyncio.gather(*self.export_tasks, return_exceptions=True)
        self.export_tasks.clear()
        
        log.info("Stopped distributed tracing export")
    
    async def _span_exporter(self):
        """Export completed spans"""
        while self.is_exporting:
            try:
                # Export spans to external systems
                # This would integrate with Jaeger, Zipkin, etc.
                
                await asyncio.sleep(10)  # Export every 10 seconds
                
            except Exception as e:
                log.error(f"Span export error: {e}")
                await asyncio.sleep(30)
    
    def get_trace_stats(self) -> Dict[str, Any]:
        """Get tracing statistics"""
        with self.lock:
            operation_stats = defaultdict(lambda: {"count": 0, "total_duration": 0, "errors": 0})
            
            for span in self.completed_spans:
                op_stats = operation_stats[span.operation_name]
                op_stats["count"] += 1
                if span.duration:
                    op_stats["total_duration"] += span.duration
                if span.status == "error":
                    op_stats["errors"] += 1
            
            # Calculate averages
            for op_name, stats in operation_stats.items():
                if stats["count"] > 0:
                    stats["avg_duration"] = stats["total_duration"] / stats["count"]
                    stats["error_rate"] = stats["errors"] / stats["count"]
            
            return {
                "active_spans": len(self.active_spans),
                "completed_spans": len(self.completed_spans),
                "operation_stats": dict(operation_stats)
            }

class AlertManager:
    """Predictive alerting system"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        
        # Alert rules
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        
        # Alert history
        self.alert_history: deque = deque(maxlen=1000)
        
        # Notification channels
        self.notification_channels: Dict[str, Callable] = {}
        
        # Background monitoring
        self.monitoring_tasks = []
        self.is_monitoring = False
        
        # Predictive analysis
        self.prediction_models: Dict[str, Any] = {}
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Setup default alert rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        self.alert_rules.update({
            "high_memory_usage": {
                "metric": "system.memory.used",
                "threshold": 0.85,  # 85% of available memory
                "operator": ">",
                "severity": AlertSeverity.WARNING,
                "duration": 300,  # 5 minutes
                "description": "System memory usage is high"
            },
            "high_cpu_usage": {
                "metric": "system.cpu.percent",
                "threshold": 80.0,
                "operator": ">",
                "severity": AlertSeverity.WARNING,
                "duration": 300,
                "description": "System CPU usage is high"
            },
            "low_cache_hit_rate": {
                "metric": "cache_hit_rate",
                "threshold": 0.5,
                "operator": "<",
                "severity": AlertSeverity.WARNING,
                "duration": 600,  # 10 minutes
                "description": "Cache hit rate is low"
            },
            "pattern_detection_errors": {
                "metric": "pattern_detection_errors",
                "threshold": 10,
                "operator": ">",
                "severity": AlertSeverity.ERROR,
                "duration": 60,  # 1 minute
                "description": "High number of pattern detection errors"
            },
            "consensus_failures": {
                "metric": "consensus_failures",
                "threshold": 5,
                "operator": ">",
                "severity": AlertSeverity.CRITICAL,
                "duration": 120,  # 2 minutes
                "description": "Consensus system experiencing failures"
            }
        })
    
    async def start_monitoring(self):
        """Start alert monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_tasks = [
            asyncio.create_task(self._alert_evaluator()),
            asyncio.create_task(self._predictive_analyzer())
        ]
        
        log.info("Started alert monitoring")
    
    async def stop_monitoring(self):
        """Stop alert monitoring"""
        self.is_monitoring = False
        
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
        
        log.info("Stopped alert monitoring")
    
    async def _alert_evaluator(self):
        """Evaluate alert rules"""
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                for rule_name, rule in self.alert_rules.items():
                    await self._evaluate_rule(rule_name, rule, current_time)
                
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
            except Exception as e:
                log.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_rule(self, rule_name: str, rule: Dict[str, Any], current_time: float):
        """Evaluate a single alert rule"""
        try:
            # Get metric statistics
            stats = self.metrics_collector.get_metric_stats(
                rule["metric"], 
                rule.get("duration", 300)
            )
            
            if "error" in stats:
                return  # Metric not available
            
            # Get value to compare
            value = stats.get("latest_value", 0)
            threshold = rule["threshold"]
            operator = rule["operator"]
            
            # Evaluate condition
            triggered = False
            if operator == ">":
                triggered = value > threshold
            elif operator == "<":
                triggered = value < threshold
            elif operator == ">=":
                triggered = value >= threshold
            elif operator == "<=":
                triggered = value <= threshold
            elif operator == "==":
                triggered = value == threshold
            elif operator == "!=":
                triggered = value != threshold
            
            # Handle alert state
            alert_key = f"{rule_name}_{rule['metric']}"
            
            if triggered:
                if alert_key not in self.active_alerts:
                    # Create new alert
                    alert = Alert(
                        name=rule_name,
                        severity=AlertSeverity(rule["severity"]),
                        message=f"{rule['description']}: {value} {operator} {threshold}",
                        source="alert_manager",
                        metadata={
                            "metric": rule["metric"],
                            "value": value,
                            "threshold": threshold,
                            "operator": operator,
                            "stats": stats
                        }
                    )
                    
                    with self.lock:
                        self.active_alerts[alert_key] = alert
                        self.alert_history.append(alert)
                    
                    # Send notification
                    await self._send_notification(alert)
                    
                    log.warning(f"Alert triggered: {alert.name} - {alert.message}")
            else:
                if alert_key in self.active_alerts:
                    # Resolve alert
                    with self.lock:
                        alert = self.active_alerts[alert_key]
                        alert.resolve()
                        del self.active_alerts[alert_key]
                    
                    log.info(f"Alert resolved: {alert.name}")
                    
        except Exception as e:
            log.error(f"Rule evaluation error for {rule_name}: {e}")
    
    async def _predictive_analyzer(self):
        """Analyze metrics for predictive alerting"""
        while self.is_monitoring:
            try:
                # Analyze trends and predict issues
                await self._analyze_trends()
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                log.error(f"Predictive analysis error: {e}")
                await asyncio.sleep(600)
    
    async def _analyze_trends(self):
        """Analyze metric trends for predictive alerting"""
        current_time = time.time()
        
        # Analyze key metrics for trends
        key_metrics = ["system.memory.used", "system.cpu.percent", "cache_hit_rate"]
        
        for metric_name in key_metrics:
            try:
                # Get recent data
                stats = self.metrics_collector.get_metric_stats(metric_name, 3600)  # 1 hour
                
                if "error" in stats:
                    continue
                
                # Simple trend analysis
                with self.metrics_collector.lock:
                    if metric_name in self.metrics_collector.time_series:
                        data_points = list(self.metrics_collector.time_series[metric_name])
                        
                        if len(data_points) >= 10:
                            # Calculate trend
                            recent_values = [value for _, value in data_points[-10:]]
                            older_values = [value for _, value in data_points[-20:-10]] if len(data_points) >= 20 else []
                            
                            if older_values:
                                recent_avg = statistics.mean(recent_values)
                                older_avg = statistics.mean(older_values)
                                trend = (recent_avg - older_avg) / older_avg if older_avg != 0 else 0
                                
                                # Check for concerning trends
                                if abs(trend) > 0.2:  # 20% change
                                    await self._create_predictive_alert(metric_name, trend, recent_avg)
                                    
            except Exception as e:
                log.debug(f"Trend analysis error for {metric_name}: {e}")
    
    async def _create_predictive_alert(self, metric_name: str, trend: float, current_value: float):
        """Create predictive alert based on trend analysis"""
        severity = AlertSeverity.INFO
        if abs(trend) > 0.5:  # 50% change
            severity = AlertSeverity.WARNING
        
        trend_direction = "increasing" if trend > 0 else "decreasing"
        
        alert = Alert(
            name=f"predictive_{metric_name.replace('.', '_')}",
            severity=severity,
            message=f"Metric {metric_name} is {trend_direction} rapidly (trend: {trend:.1%})",
            source="predictive_analyzer",
            metadata={
                "metric": metric_name,
                "trend": trend,
                "current_value": current_value,
                "prediction_type": "trend_analysis"
            }
        )
        
        alert_key = f"predictive_{metric_name}"
        
        with self.lock:
            # Only create if not already active
            if alert_key not in self.active_alerts:
                self.active_alerts[alert_key] = alert
                self.alert_history.append(alert)
                
                # Send notification
                await self._send_notification(alert)
                
                log.info(f"Predictive alert created: {alert.name}")
    
    async def _send_notification(self, alert: Alert):
        """Send alert notification"""
        for channel_name, channel_func in self.notification_channels.items():
            try:
                if asyncio.iscoroutinefunction(channel_func):
                    await channel_func(alert)
                else:
                    channel_func(alert)
            except Exception as e:
                log.error(f"Notification failed for channel {channel_name}: {e}")
    
    def add_alert_rule(self, name: str, metric: str, threshold: float, operator: str,
                      severity: AlertSeverity = AlertSeverity.WARNING, duration: int = 300,
                      description: str = "") -> bool:
        """Add custom alert rule"""
        try:
            self.alert_rules[name] = {
                "metric": metric,
                "threshold": threshold,
                "operator": operator,
                "severity": severity,
                "duration": duration,
                "description": description or f"Alert for {metric}"
            }
            log.info(f"Added alert rule: {name}")
            return True
        except Exception as e:
            log.error(f"Failed to add alert rule {name}: {e}")
            return False
    
    def add_notification_channel(self, name: str, handler: Callable) -> bool:
        """Add notification channel"""
        try:
            self.notification_channels[name] = handler
            log.info(f"Added notification channel: {name}")
            return True
        except Exception as e:
            log.error(f"Failed to add notification channel {name}: {e}")
            return False
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alerting statistics"""
        with self.lock:
            severity_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                severity_counts[alert.severity.value] += 1
            
            return {
                "active_alerts": len(self.active_alerts),
                "total_rules": len(self.alert_rules),
                "alert_history_size": len(self.alert_history),
                "severity_breakdown": dict(severity_counts),
                "notification_channels": len(self.notification_channels)
            }

class ObservabilityManager:
    """Main observability manager coordinating all monitoring components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.metrics_collector = MetricsCollector(config.get("service_name", "neural-mesh"))
        self.distributed_tracer = DistributedTracer(config.get("service_name", "neural-mesh"))
        self.alert_manager = AlertManager(self.metrics_collector)
        
        # Rate limiters
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self._setup_default_rate_limiters()
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
        
        # Health check endpoints
        self.health_checks: Dict[str, Callable] = {}
        
        # Performance tracking
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def _setup_default_rate_limiters(self):
        """Setup default rate limiters"""
        self.rate_limiters.update({
            "memory_operations": RateLimiter(
                RateLimitStrategy.TOKEN_BUCKET, 
                limit=1000, 
                window_seconds=60.0
            ),
            "pattern_detection": RateLimiter(
                RateLimitStrategy.SLIDING_WINDOW,
                limit=100,
                window_seconds=60.0
            ),
            "knowledge_synthesis": RateLimiter(
                RateLimitStrategy.LEAKY_BUCKET,
                limit=50,
                window_seconds=60.0
            ),
            "api_requests": RateLimiter(
                RateLimitStrategy.FIXED_WINDOW,
                limit=10000,
                window_seconds=3600.0  # Per hour
            )
        })
    
    async def initialize(self):
        """Initialize observability manager"""
        log.info("Initializing observability manager")
        
        # Start components
        await self.metrics_collector.start_collection()
        await self.distributed_tracer.start_export()
        await self.alert_manager.start_monitoring()
        
        # Start background tasks
        self.is_running = True
        self.background_tasks = [
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._performance_analyzer())
        ]
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        log.info("Observability manager initialized")
    
    async def shutdown(self):
        """Shutdown observability manager"""
        log.info("Shutting down observability manager")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        # Stop components
        await self.metrics_collector.stop_collection()
        await self.distributed_tracer.stop_export()
        await self.alert_manager.stop_monitoring()
        
        log.info("Observability manager shutdown complete")
    
    def _setup_default_health_checks(self):
        """Setup default health check endpoints"""
        self.health_checks.update({
            "memory_usage": self._check_memory_health,
            "cpu_usage": self._check_cpu_health,
            "cache_performance": self._check_cache_health,
            "error_rates": self._check_error_rates
        })
    
    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory health"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            status = "healthy"
            if memory.percent > 90:
                status = "critical"
            elif memory.percent > 80:
                status = "warning"
            
            return {
                "status": status,
                "memory_percent": memory.percent,
                "available_gb": memory.available / (1024**3)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_cpu_health(self) -> Dict[str, Any]:
        """Check CPU health"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            
            status = "healthy"
            if cpu_percent > 90:
                status = "critical"
            elif cpu_percent > 80:
                status = "warning"
            
            return {
                "status": status,
                "cpu_percent": cpu_percent
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache performance health"""
        # This would integrate with actual cache systems
        return {"status": "healthy", "note": "Cache health check not implemented"}
    
    def _check_error_rates(self) -> Dict[str, Any]:
        """Check error rates"""
        with self.lock:
            total_requests = sum(self.request_counts.values())
            total_errors = sum(self.error_counts.values())
            
            if total_requests == 0:
                error_rate = 0.0
            else:
                error_rate = total_errors / total_requests
            
            status = "healthy"
            if error_rate > 0.1:  # 10% error rate
                status = "critical"
            elif error_rate > 0.05:  # 5% error rate
                status = "warning"
            
            return {
                "status": status,
                "error_rate": error_rate,
                "total_requests": total_requests,
                "total_errors": total_errors
            }
    
    async def _health_monitor(self):
        """Background health monitoring"""
        while self.is_running:
            try:
                # Run all health checks
                health_results = {}
                for check_name, check_func in self.health_checks.items():
                    try:
                        result = check_func()
                        health_results[check_name] = result
                    except Exception as e:
                        health_results[check_name] = {"status": "error", "error": str(e)}
                
                # Log critical health issues
                for check_name, result in health_results.items():
                    if result.get("status") == "critical":
                        log.error(f"Critical health issue in {check_name}: {result}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                log.error(f"Health monitoring error: {e}")
                await asyncio.sleep(120)
    
    async def _performance_analyzer(self):
        """Background performance analysis"""
        while self.is_running:
            try:
                # Analyze response times
                with self.lock:
                    for endpoint, times in self.response_times.items():
                        if len(times) >= 10:
                            avg_time = statistics.mean(times)
                            p95_time = statistics.quantiles(times, n=20)[18]  # 95th percentile
                            
                            # Record metrics
                            self.metrics_collector.record_metric(
                                f"response_time_avg_{endpoint.replace('/', '_')}",
                                avg_time,
                                MetricType.GAUGE
                            )
                            
                            self.metrics_collector.record_metric(
                                f"response_time_p95_{endpoint.replace('/', '_')}",
                                p95_time,
                                MetricType.GAUGE
                            )
                            
                            # Clear old data
                            self.response_times[endpoint] = times[-100:]  # Keep last 100
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                log.error(f"Performance analysis error: {e}")
                await asyncio.sleep(600)
    
    # Rate limiting
    def check_rate_limit(self, limiter_name: str, tokens: int = 1) -> bool:
        """Check if request is within rate limit"""
        rate_limiter = self.rate_limiters.get(limiter_name)
        if not rate_limiter:
            return True  # No limit configured
        
        return rate_limiter.is_allowed(tokens)
    
    def get_rate_limit_stats(self, limiter_name: str) -> Dict[str, Any]:
        """Get rate limit statistics"""
        rate_limiter = self.rate_limiters.get(limiter_name)
        if not rate_limiter:
            return {"error": "Rate limiter not found"}
        
        return rate_limiter.get_stats()
    
    # Tracing
    def start_trace(self, operation_name: str, **tags) -> TraceSpan:
        """Start distributed trace"""
        return self.distributed_tracer.start_span(operation_name, tags=tags)
    
    def finish_trace(self, span: TraceSpan, status: str = "ok"):
        """Finish distributed trace"""
        self.distributed_tracer.finish_span(span, status)
    
    # Metrics
    def record_request(self, endpoint: str, response_time: float, status_code: int):
        """Record request metrics"""
        with self.lock:
            self.request_counts[endpoint] += 1
            self.response_times[endpoint].append(response_time)
            
            if status_code >= 400:
                self.error_counts[endpoint] += 1
        
        # Record in metrics collector
        self.metrics_collector.record_metric(
            "http_requests_total",
            1,
            MetricType.COUNTER,
            {"endpoint": endpoint, "status_code": str(status_code)}
        )
        
        self.metrics_collector.record_metric(
            "http_request_duration_seconds",
            response_time,
            MetricType.HISTOGRAM,
            {"endpoint": endpoint}
        )
    
    # Health checks
    def add_health_check(self, name: str, check_func: Callable) -> bool:
        """Add custom health check"""
        try:
            self.health_checks[name] = check_func
            log.info(f"Added health check: {name}")
            return True
        except Exception as e:
            log.error(f"Failed to add health check {name}: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        health_results = {}
        overall_status = "healthy"
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = check_func()
                health_results[check_name] = result
                
                if result.get("status") == "critical":
                    overall_status = "critical"
                elif result.get("status") == "warning" and overall_status == "healthy":
                    overall_status = "warning"
                    
            except Exception as e:
                health_results[check_name] = {"status": "error", "error": str(e)}
                if overall_status == "healthy":
                    overall_status = "warning"
        
        return {
            "overall_status": overall_status,
            "checks": health_results,
            "timestamp": time.time()
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive observability statistics"""
        return {
            "metrics": {
                "custom_metrics": len(self.metrics_collector.custom_metrics),
                "time_series": len(self.metrics_collector.time_series)
            },
            "tracing": self.distributed_tracer.get_trace_stats(),
            "alerts": self.alert_manager.get_alert_stats(),
            "rate_limits": {
                name: limiter.get_stats() 
                for name, limiter in self.rate_limiters.items()
            },
            "health": self.get_health_status(),
            "performance": {
                "total_requests": sum(self.request_counts.values()),
                "total_errors": sum(self.error_counts.values()),
                "endpoints_monitored": len(self.request_counts)
            }
        }
