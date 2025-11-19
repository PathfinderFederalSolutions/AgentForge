"""
Enhanced Observability System - Million-Scale Monitoring and Alerting
Comprehensive monitoring, alerting, and observability for AGI-scale operations
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import psutil
import os

# Metrics imports (graceful degradation)
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    Counter = Histogram = Gauge = Summary = Info = lambda *args, **kwargs: None

log = logging.getLogger("enhanced-observability")

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Alert:
    """Alert definition"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str  # PromQL expression
    duration: str  # How long condition must be true
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    def to_prometheus_rule(self) -> Dict[str, Any]:
        """Convert to Prometheus alerting rule format"""
        return {
            "alert": self.name,
            "expr": self.condition,
            "for": self.duration,
            "labels": {
                **self.labels,
                "severity": self.severity.value,
                "alert_id": self.id
            },
            "annotations": {
                **self.annotations,
                "summary": self.description,
                "description": self.description
            }
        }

@dataclass
class Dashboard:
    """Grafana dashboard definition"""
    id: str
    title: str
    description: str
    panels: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    refresh: str = "30s"
    
    def add_panel(self, panel_config: Dict[str, Any]):
        """Add panel to dashboard"""
        self.panels.append(panel_config)
    
    def to_grafana_json(self) -> Dict[str, Any]:
        """Convert to Grafana dashboard JSON"""
        return {
            "id": None,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "refresh": self.refresh,
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "panels": self.panels,
            "schemaVersion": 16,
            "version": 1
        }

class EnhancedObservabilitySystem:
    """Comprehensive observability system for million-scale operations"""
    
    def __init__(self):
        # Core components
        self.registry = CollectorRegistry() if METRICS_AVAILABLE else None
        self.custom_metrics: Dict[str, Any] = {}
        self.alerts: Dict[str, Alert] = {}
        self.dashboards: Dict[str, Dashboard] = {}
        
        # System monitoring
        self.system_metrics = {}
        self.health_checks: Dict[str, Callable] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        # Initialize core metrics and alerts
        self._initialize_core_metrics()
        self._initialize_core_alerts()
        self._initialize_core_dashboards()
        
        log.info("Enhanced observability system initialized")
    
    def _initialize_core_metrics(self):
        """Initialize core system metrics"""
        if not METRICS_AVAILABLE:
            return
        
        # System resource metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        self.system_memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        self.system_disk_usage = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            ['mountpoint'],
            registry=self.registry
        )
        
        # AgentForge specific metrics
        self.agentforge_agents_active = Gauge(
            'agentforge_agents_active',
            'Number of active agents',
            ['swarm_id'],
            registry=self.registry
        )
        self.agentforge_memory_operations = Counter(
            'agentforge_memory_operations_total',
            'Total memory operations',
            ['tier', 'operation', 'status'],
            registry=self.registry
        )
        self.agentforge_task_queue_depth = Gauge(
            'agentforge_task_queue_depth',
            'Task queue depth',
            ['priority', 'strategy'],
            registry=self.registry
        )
        
        # Million-scale specific metrics
        self.million_scale_throughput = Gauge(
            'agentforge_million_scale_throughput_ops_per_sec',
            'Million-scale operations throughput',
            ['component'],
            registry=self.registry
        )
        self.million_scale_latency = Histogram(
            'agentforge_million_scale_latency_seconds',
            'Million-scale operation latency',
            ['component', 'operation'],
            registry=self.registry
        )
    
    def _initialize_core_alerts(self):
        """Initialize core alerting rules"""
        # System resource alerts
        self.alerts["high_cpu_usage"] = Alert(
            id="agentforge_high_cpu",
            name="AgentForgeHighCPUUsage",
            description="AgentForge system CPU usage is high",
            severity=AlertSeverity.WARNING,
            condition="system_cpu_usage_percent > 80",
            duration="5m",
            labels={"component": "system", "team": "platform"},
            annotations={
                "runbook": "https://docs.agentforge.io/runbooks/high-cpu",
                "impact": "Performance degradation possible"
            }
        )
        
        self.alerts["high_memory_usage"] = Alert(
            id="agentforge_high_memory",
            name="AgentForgeHighMemoryUsage", 
            description="AgentForge system memory usage is high",
            severity=AlertSeverity.WARNING,
            condition="system_memory_usage_percent > 85",
            duration="3m",
            labels={"component": "system", "team": "platform"},
            annotations={
                "runbook": "https://docs.agentforge.io/runbooks/high-memory",
                "impact": "System stability at risk"
            }
        )
        
        # Million-scale specific alerts
        self.alerts["million_scale_throughput_low"] = Alert(
            id="agentforge_throughput_low",
            name="AgentForgeMillionScaleThroughputLow",
            description="Million-scale throughput below target",
            severity=AlertSeverity.WARNING,
            condition="agentforge_million_scale_throughput_ops_per_sec < 1000",
            duration="10m",
            labels={"component": "orchestration", "team": "platform"},
            annotations={
                "runbook": "https://docs.agentforge.io/runbooks/throughput-optimization",
                "target": "1000+ operations/second for million-scale readiness"
            }
        )
        
        self.alerts["task_queue_backlog"] = Alert(
            id="agentforge_task_backlog",
            name="AgentForgeTaskQueueBacklog",
            description="High task queue backlog detected",
            severity=AlertSeverity.CRITICAL,
            condition="sum(agentforge_task_queue_depth) > 5000",
            duration="5m",
            labels={"component": "orchestration", "team": "platform"},
            annotations={
                "runbook": "https://docs.agentforge.io/runbooks/task-backlog",
                "impact": "Task processing delays, potential system overload"
            }
        )
        
        # Security alerts
        self.alerts["security_incidents"] = Alert(
            id="agentforge_security_incidents",
            name="AgentForgeSecurityIncidents",
            description="Security incidents detected",
            severity=AlertSeverity.CRITICAL,
            condition="increase(agentforge_security_incidents_total[15m]) > 0",
            duration="1m",
            labels={"component": "security", "team": "security"},
            annotations={
                "runbook": "https://docs.agentforge.io/runbooks/security-incident",
                "impact": "Potential security breach"
            }
        )
    
    def _initialize_core_dashboards(self):
        """Initialize core Grafana dashboards"""
        # Million-Scale Operations Dashboard
        million_scale_dashboard = Dashboard(
            id="agentforge_million_scale",
            title="AgentForge Million-Scale Operations",
            description="Comprehensive monitoring for million-scale agent coordination",
            tags=["agentforge", "million-scale", "operations"]
        )
        
        # Add panels for million-scale dashboard
        million_scale_dashboard.add_panel({
            "id": 1,
            "title": "Agent Throughput",
            "type": "stat",
            "targets": [{
                "expr": "sum(rate(agentforge_agents_active[5m]))",
                "legendFormat": "Agents/sec"
            }],
            "fieldConfig": {
                "defaults": {
                    "unit": "ops",
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": 100},
                            {"color": "green", "value": 1000}
                        ]
                    }
                }
            }
        })
        
        million_scale_dashboard.add_panel({
            "id": 2,
            "title": "Memory Operations Rate",
            "type": "graph",
            "targets": [
                {
                    "expr": "sum(rate(agentforge_memory_operations_total[5m])) by (tier)",
                    "legendFormat": "{{tier}}"
                }
            ],
            "yAxes": [{"unit": "ops"}]
        })
        
        million_scale_dashboard.add_panel({
            "id": 3,
            "title": "Task Queue Depths",
            "type": "graph",
            "targets": [
                {
                    "expr": "agentforge_task_queue_depth",
                    "legendFormat": "{{priority}} - {{strategy}}"
                }
            ],
            "yAxes": [{"unit": "short"}]
        })
        
        million_scale_dashboard.add_panel({
            "id": 4,
            "title": "System Resources",
            "type": "graph",
            "targets": [
                {"expr": "system_cpu_usage_percent", "legendFormat": "CPU %"},
                {"expr": "system_memory_usage_percent", "legendFormat": "Memory %"}
            ],
            "yAxes": [{"unit": "percent", "max": 100}]
        })
        
        self.dashboards["million_scale"] = million_scale_dashboard
        
        # Neural Mesh Memory Dashboard
        memory_dashboard = Dashboard(
            id="agentforge_neural_mesh",
            title="AgentForge Neural Mesh Memory",
            description="Neural mesh memory system monitoring",
            tags=["agentforge", "neural-mesh", "memory"]
        )
        
        memory_dashboard.add_panel({
            "id": 1,
            "title": "Memory Tier Utilization",
            "type": "graph",
            "targets": [
                {
                    "expr": "agentforge_memory_operations_total",
                    "legendFormat": "{{tier}} - {{operation}}"
                }
            ]
        })
        
        memory_dashboard.add_panel({
            "id": 2,
            "title": "Embedding Generation Rate",
            "type": "stat",
            "targets": [{
                "expr": "rate(agentforge_embeddings_generated_total[5m])",
                "legendFormat": "Embeddings/sec"
            }]
        })
        
        self.dashboards["neural_mesh"] = memory_dashboard
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        log.info("Starting enhanced observability monitoring")
        
        # Start background monitoring tasks
        monitoring_tasks = [
            asyncio.create_task(self._system_metrics_loop()),
            asyncio.create_task(self._performance_analysis_loop()),
            asyncio.create_task(self._health_check_loop())
        ]
        
        # Store task references
        self._monitoring_tasks = monitoring_tasks
        
        log.info("Enhanced observability monitoring started")
    
    async def _system_metrics_loop(self):
        """Collect system metrics continuously"""
        while True:
            try:
                if METRICS_AVAILABLE:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.system_cpu_usage.set(cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.system_memory_usage.set(memory.percent)
                    
                    # Disk usage
                    for partition in psutil.disk_partitions():
                        try:
                            usage = psutil.disk_usage(partition.mountpoint)
                            self.system_disk_usage.labels(
                                mountpoint=partition.mountpoint
                            ).set(usage.percent)
                        except (PermissionError, FileNotFoundError):
                            pass  # Skip inaccessible partitions
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                log.error(f"System metrics collection error: {e}")
                await asyncio.sleep(10)
    
    async def _performance_analysis_loop(self):
        """Analyze performance trends"""
        while True:
            try:
                # Collect current performance snapshot
                performance_snapshot = {
                    "timestamp": time.time(),
                    "cpu_usage": psutil.cpu_percent(),
                    "memory_usage": psutil.virtual_memory().percent,
                    "active_connections": len(psutil.net_connections()),
                    "process_count": len(psutil.pids())
                }
                
                # Add to history
                self.performance_history.append(performance_snapshot)
                
                # Keep last 24 hours (assuming 5-minute intervals)
                if len(self.performance_history) > 288:
                    self.performance_history.pop(0)
                
                # Analyze trends
                if len(self.performance_history) >= 12:  # At least 1 hour of data
                    await self._analyze_performance_trends()
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                log.error(f"Performance analysis error: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_performance_trends(self):
        """Analyze performance trends and generate insights"""
        if len(self.performance_history) < 12:
            return
        
        # Calculate trends over last hour
        recent_data = self.performance_history[-12:]
        
        # CPU trend
        cpu_values = [snapshot["cpu_usage"] for snapshot in recent_data]
        cpu_trend = (cpu_values[-1] - cpu_values[0]) / len(cpu_values)
        
        # Memory trend
        memory_values = [snapshot["memory_usage"] for snapshot in recent_data]
        memory_trend = (memory_values[-1] - memory_values[0]) / len(memory_values)
        
        # Generate insights
        insights = []
        if cpu_trend > 5:  # CPU increasing by >5% per interval
            insights.append("CPU usage trending upward - consider scaling")
        if memory_trend > 3:  # Memory increasing by >3% per interval
            insights.append("Memory usage trending upward - monitor for leaks")
        
        if insights:
            log.info(f"Performance insights: {'; '.join(insights)}")
    
    async def _health_check_loop(self):
        """Perform comprehensive health checks"""
        while True:
            try:
                # Run all registered health checks
                health_results = {}
                
                for check_name, check_func in self.health_checks.items():
                    try:
                        result = await check_func()
                        health_results[check_name] = result
                    except Exception as e:
                        health_results[check_name] = {
                            "status": "error",
                            "error": str(e)
                        }
                
                # Log health status
                unhealthy_checks = [
                    name for name, result in health_results.items()
                    if result.get("status") != "healthy"
                ]
                
                if unhealthy_checks:
                    log.warning(f"Unhealthy components: {unhealthy_checks}")
                
                await asyncio.sleep(60)  # Health check every minute
                
            except Exception as e:
                log.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)
    
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self.health_checks[name] = check_func
        log.info(f"Registered health check: {name}")
    
    def register_custom_metric(self, name: str, metric_type: MetricType, 
                             description: str, labels: List[str] = None):
        """Register a custom metric"""
        if not METRICS_AVAILABLE:
            return None
        
        labels = labels or []
        
        if metric_type == MetricType.COUNTER:
            metric = Counter(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.GAUGE:
            metric = Gauge(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.HISTOGRAM:
            metric = Histogram(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.SUMMARY:
            metric = Summary(name, description, labels, registry=self.registry)
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        
        self.custom_metrics[name] = metric
        log.info(f"Registered custom metric: {name} ({metric_type.value})")
        return metric
    
    def add_alert(self, alert: Alert):
        """Add alerting rule"""
        self.alerts[alert.id] = alert
        log.info(f"Added alert: {alert.name}")
    
    def add_dashboard(self, dashboard: Dashboard):
        """Add Grafana dashboard"""
        self.dashboards[dashboard.id] = dashboard
        log.info(f"Added dashboard: {dashboard.title}")
    
    def get_prometheus_config(self) -> Dict[str, Any]:
        """Generate Prometheus configuration"""
        # Generate alerting rules
        alerting_rules = {
            "groups": [
                {
                    "name": "agentforge_enhanced_alerts",
                    "interval": "30s",
                    "rules": [
                        alert.to_prometheus_rule() 
                        for alert in self.alerts.values() 
                        if alert.enabled
                    ]
                }
            ]
        }
        
        # Generate recording rules for performance
        recording_rules = {
            "groups": [
                {
                    "name": "agentforge_enhanced_recording",
                    "interval": "15s",
                    "rules": [
                        {
                            "record": "agentforge:million_scale_ops_rate",
                            "expr": "sum(rate(agentforge_million_scale_throughput_ops_per_sec[5m]))"
                        },
                        {
                            "record": "agentforge:memory_efficiency",
                            "expr": "sum(rate(agentforge_memory_operations_total{status=\"success\"}[5m])) / sum(rate(agentforge_memory_operations_total[5m]))"
                        },
                        {
                            "record": "agentforge:system_health_score",
                            "expr": "(100 - system_cpu_usage_percent) * (100 - system_memory_usage_percent) / 100"
                        }
                    ]
                }
            ]
        }
        
        return {
            "alerting_rules": alerting_rules,
            "recording_rules": recording_rules
        }
    
    def get_grafana_dashboards(self) -> Dict[str, Dict[str, Any]]:
        """Get all Grafana dashboards"""
        return {
            dashboard_id: dashboard.to_grafana_json()
            for dashboard_id, dashboard in self.dashboards.items()
        }
    
    def get_metrics_endpoint(self) -> str:
        """Get Prometheus metrics in text format"""
        if not METRICS_AVAILABLE or not self.registry:
            return "# Metrics not available\n"
        
        return generate_latest(self.registry).decode('utf-8')
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        try:
            # System resources
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Process info
            process_count = len(psutil.pids())
            
            # Health check results
            health_results = {}
            for check_name, check_func in self.health_checks.items():
                try:
                    health_results[check_name] = await check_func()
                except Exception as e:
                    health_results[check_name] = {"status": "error", "error": str(e)}
            
            return {
                "timestamp": time.time(),
                "system_resources": {
                    "cpu_usage_percent": cpu_percent,
                    "memory_usage_percent": memory.percent,
                    "disk_usage_percent": (disk.used / disk.total) * 100,
                    "available_memory_gb": memory.available / (1024**3),
                    "available_disk_gb": disk.free / (1024**3)
                },
                "network_stats": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_received": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_received": network.packets_recv
                },
                "process_stats": {
                    "total_processes": process_count
                },
                "health_checks": health_results,
                "alerts_configured": len(self.alerts),
                "dashboards_configured": len(self.dashboards),
                "performance_history_points": len(self.performance_history)
            }
            
        except Exception as e:
            log.error(f"System overview generation failed: {e}")
            return {"error": str(e), "timestamp": time.time()}

# Global observability instance
enhanced_observability: Optional[EnhancedObservabilitySystem] = None

def get_enhanced_observability() -> EnhancedObservabilitySystem:
    """Get or create the global enhanced observability system"""
    global enhanced_observability
    if enhanced_observability is None:
        enhanced_observability = EnhancedObservabilitySystem()
    return enhanced_observability

async def initialize_enhanced_monitoring():
    """Initialize and start enhanced monitoring"""
    observability = get_enhanced_observability()
    await observability.start_monitoring()
    return observability
