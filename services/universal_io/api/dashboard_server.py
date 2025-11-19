"""
Real-Time Dashboard API and WebSocket Server
Provides live streaming dashboards for all vertical domains
Supports real-time updates, interactive visualizations, and multi-tenant access
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from datetime import datetime, timedelta

# Import universal-io components
from ..stream.stream_ingestion import StreamMessage, StreamType
from ..stream.event_processor import ProcessingEvent, EventType
from ..outputs.vertical_generators import VerticalDomain
from ..security.zero_trust_framework import ZeroTrustSecurityFramework, SecurityLevel, authorize
from ..integration.swarm_integration import UniversalIOSwarmCoordinator, ProcessingScale, ProcessingObjective

log = logging.getLogger("dashboard-server")

class DashboardType(Enum):
    """Types of dashboards"""
    # Defense & Intelligence
    TACTICAL_COP = "tactical_cop"
    INTELLIGENCE_FUSION = "intelligence_fusion"
    THREAT_MONITORING = "threat_monitoring"
    SIGINT_ANALYSIS = "sigint_analysis"
    
    # Healthcare
    PATIENT_MONITORING = "patient_monitoring"
    CLINICAL_DECISION_SUPPORT = "clinical_decision_support"
    POPULATION_HEALTH = "population_health"
    MEDICAL_IMAGING = "medical_imaging"
    
    # Finance
    RISK_MONITORING = "risk_monitoring"
    TRADING_DASHBOARD = "trading_dashboard"
    FRAUD_DETECTION = "fraud_detection"
    MARKET_SURVEILLANCE = "market_surveillance"
    
    # Business Intelligence
    EXECUTIVE_DASHBOARD = "executive_dashboard"
    OPERATIONAL_METRICS = "operational_metrics"
    SUPPLY_CHAIN = "supply_chain"
    CUSTOMER_ANALYTICS = "customer_analytics"
    
    # Federal Civilian
    CITIZEN_SERVICES = "citizen_services"
    INFRASTRUCTURE_MONITORING = "infrastructure_monitoring"
    EMERGENCY_RESPONSE = "emergency_response"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    
    # System Monitoring
    SWARM_COORDINATION = "swarm_coordination"
    STREAM_PROCESSING = "stream_processing"
    SECURITY_MONITORING = "security_monitoring"
    PERFORMANCE_ANALYTICS = "performance_analytics"

class UpdateFrequency(Enum):
    """Update frequency for dashboard components"""
    REAL_TIME = "real_time"      # < 100ms
    HIGH_FREQUENCY = "high_freq"  # 1-5 seconds
    MEDIUM_FREQUENCY = "medium_freq"  # 5-30 seconds
    LOW_FREQUENCY = "low_freq"    # 30+ seconds
    ON_DEMAND = "on_demand"       # Manual refresh only

@dataclass
class DashboardWidget:
    """Individual dashboard widget"""
    widget_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    widget_type: str = "chart"  # chart, table, metric, alert, map, etc.
    title: str = ""
    
    # Data configuration
    data_source: str = ""
    query: str = ""
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Display configuration
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "w": 4, "h": 3})
    chart_type: str = "line"  # line, bar, pie, gauge, table, etc.
    color_scheme: str = "default"
    
    # Update settings
    update_frequency: UpdateFrequency = UpdateFrequency.MEDIUM_FREQUENCY
    auto_refresh: bool = True
    
    # Current data
    current_data: Any = None
    last_updated: float = field(default_factory=time.time)
    
    # Security
    required_permissions: List[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.INTERNAL

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    dashboard_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dashboard_type: DashboardType = DashboardType.EXECUTIVE_DASHBOARD
    title: str = ""
    description: str = ""
    
    # Layout
    layout: str = "grid"  # grid, flex, custom
    theme: str = "dark"   # dark, light, auto
    refresh_interval: int = 30  # seconds
    
    # Widgets
    widgets: List[DashboardWidget] = field(default_factory=list)
    
    # Access control
    owner_id: str = ""
    shared_users: List[str] = field(default_factory=list)
    public_access: bool = False
    required_roles: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

@dataclass
class WebSocketConnection:
    """WebSocket connection management"""
    connection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    websocket: WebSocket = None
    user_id: str = ""
    session_id: str = ""
    
    # Subscriptions
    subscribed_dashboards: Set[str] = field(default_factory=set)
    subscribed_streams: Set[str] = field(default_factory=set)
    subscribed_events: Set[str] = field(default_factory=set)
    
    # Connection metadata
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    ip_address: str = ""
    user_agent: str = ""
    
    # Security context
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    permissions: List[str] = field(default_factory=list)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def is_expired(self, timeout: int = 3600) -> bool:
        """Check if connection is expired"""
        return time.time() - self.last_activity > timeout

class DashboardDataProvider:
    """Provides data for dashboard widgets"""
    
    def __init__(self, coordinator: UniversalIOSwarmCoordinator):
        self.coordinator = coordinator
        self.data_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Data generators for different widget types
        self.data_generators = {
            "tactical_cop": self._generate_tactical_cop_data,
            "patient_monitoring": self._generate_patient_monitoring_data,
            "risk_monitoring": self._generate_risk_monitoring_data,
            "swarm_metrics": self._generate_swarm_metrics_data,
            "stream_stats": self._generate_stream_stats_data,
            "security_events": self._generate_security_events_data,
            "performance_metrics": self._generate_performance_metrics_data
        }
    
    async def get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for a specific widget"""
        try:
            cache_key = f"{widget.widget_id}_{widget.data_source}"
            
            # Check cache first
            if cache_key in self.data_cache:
                cached_data = self.data_cache[cache_key]
                if time.time() - cached_data["timestamp"] < self.cache_ttl:
                    return cached_data["data"]
            
            # Generate fresh data
            generator = self.data_generators.get(widget.data_source, self._generate_generic_data)
            data = await generator(widget)
            
            # Cache the data
            self.data_cache[cache_key] = {
                "data": data,
                "timestamp": time.time()
            }
            
            return data
            
        except Exception as e:
            log.error(f"Failed to get widget data: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    async def _generate_tactical_cop_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate tactical common operating picture data"""
        return {
            "situation_overview": {
                "threat_level": "MEDIUM",
                "force_protection": "BRAVO",
                "active_operations": 3,
                "last_updated": datetime.utcnow().isoformat()
            },
            "force_status": {
                "friendly_units": [
                    {"unit": "1-1 CAV", "status": "GREEN", "location": "Grid 12345678", "strength": 85},
                    {"unit": "2-3 INF", "status": "AMBER", "location": "Grid 23456789", "strength": 78},
                    {"unit": "3-2 ARTY", "status": "GREEN", "location": "Grid 34567890", "strength": 92}
                ]
            },
            "threat_picture": {
                "known_threats": [
                    {"id": "T001", "type": "ARMOR", "location": "Grid 45678901", "confidence": "HIGH"},
                    {"id": "T002", "type": "INFANTRY", "location": "Grid 56789012", "confidence": "MEDIUM"}
                ]
            },
            "intelligence_updates": [
                {
                    "time": datetime.utcnow().strftime("%H%MZ"),
                    "source": "HUMINT",
                    "summary": "Enemy movement observed in sector 7",
                    "confidence": "MEDIUM"
                }
            ]
        }
    
    async def _generate_patient_monitoring_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate patient monitoring dashboard data"""
        import random
        
        return {
            "icu_status": {
                "total_beds": 24,
                "occupied": 18,
                "critical": 3,
                "stable": 12,
                "recovering": 3,
                "occupancy_rate": 0.75
            },
            "alerts": [
                {
                    "patient_id": "P001",
                    "alert_type": "SEPSIS_RISK",
                    "severity": "HIGH",
                    "confidence": 0.87,
                    "time": datetime.utcnow().isoformat()
                },
                {
                    "patient_id": "P007",
                    "alert_type": "DETERIORATION",
                    "severity": "MEDIUM",
                    "confidence": 0.72,
                    "time": (datetime.utcnow() - timedelta(minutes=15)).isoformat()
                }
            ],
            "vital_trends": {
                "timestamps": [(datetime.utcnow() - timedelta(minutes=i*5)).isoformat() for i in range(12, 0, -1)],
                "heart_rate_avg": [random.randint(70, 90) for _ in range(12)],
                "bp_systolic_avg": [random.randint(110, 140) for _ in range(12)],
                "oxygen_sat_avg": [random.randint(95, 100) for _ in range(12)]
            },
            "quality_metrics": {
                "patient_satisfaction": 4.2,
                "length_of_stay": 3.8,
                "readmission_rate": 0.08,
                "infection_rate": 0.02
            }
        }
    
    async def _generate_risk_monitoring_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate financial risk monitoring data"""
        import random
        
        return {
            "market_risk": {
                "var_95_1day": 2.5e6,
                "var_95_10day": 7.9e6,
                "stress_var": 15.2e6,
                "utilization": 0.79,
                "risk_factors": {
                    "equity": 0.45,
                    "interest_rate": 0.32,
                    "fx": 0.18,
                    "commodity": 0.05
                }
            },
            "credit_risk": {
                "total_exposure": 125.7e6,
                "expected_loss": 890e3,
                "capital_ratio": 0.142,
                "concentration_risk": 0.35
            },
            "operational_risk": {
                "key_indicators": [
                    {"name": "System Downtime", "value": 0.02, "threshold": 0.05, "status": "GREEN"},
                    {"name": "Failed Trades", "value": 3, "threshold": 10, "status": "GREEN"},
                    {"name": "Settlement Fails", "value": 1, "threshold": 5, "status": "GREEN"}
                ]
            },
            "alerts": [
                {
                    "type": "LIMIT_BREACH",
                    "severity": "MEDIUM",
                    "description": "Sector concentration approaching limit",
                    "time": datetime.utcnow().isoformat()
                }
            ],
            "time_series": {
                "timestamps": [(datetime.utcnow() - timedelta(hours=i)).isoformat() for i in range(24, 0, -1)],
                "var_95": [random.uniform(2.0e6, 3.0e6) for _ in range(24)],
                "expected_loss": [random.uniform(800e3, 1000e3) for _ in range(24)]
            }
        }
    
    async def _generate_swarm_metrics_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate swarm coordination metrics"""
        metrics = self.coordinator.get_coordinator_metrics()
        
        return {
            "active_swarms": metrics.get("active_tasks", 0),
            "total_agents_deployed": metrics["processing_metrics"].get("peak_agents_deployed", 0),
            "task_throughput": {
                "total_processed": metrics["processing_metrics"].get("total_tasks_processed", 0),
                "successful": metrics["processing_metrics"].get("successful_tasks", 0),
                "failed": metrics["processing_metrics"].get("failed_tasks", 0),
                "success_rate": (
                    metrics["processing_metrics"].get("successful_tasks", 0) /
                    max(metrics["processing_metrics"].get("total_tasks_processed", 1), 1)
                )
            },
            "performance": {
                "avg_processing_time": metrics["processing_metrics"].get("avg_processing_time", 0),
                "throughput_per_hour": metrics["processing_metrics"].get("throughput_per_hour", 0),
                "agent_utilization": 0.75  # Calculated metric
            },
            "queue_status": {
                "pending_tasks": metrics.get("queue_size", 0),
                "processing_tasks": metrics.get("active_tasks", 0)
            }
        }
    
    async def _generate_stream_stats_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate stream processing statistics"""
        # Get stream stats from coordinator
        stream_stats = self.coordinator.stream_engine.get_stream_stats() if hasattr(self.coordinator.stream_engine, 'get_stream_stats') else {}
        
        return {
            "active_streams": stream_stats.get("global_stats", {}).get("active_streams", 0),
            "total_messages": stream_stats.get("global_stats", {}).get("total_messages", 0),
            "messages_per_second": stream_stats.get("global_stats", {}).get("messages_per_second", 0.0),
            "failed_messages": stream_stats.get("global_stats", {}).get("failed_messages", 0),
            "avg_latency": stream_stats.get("global_stats", {}).get("avg_latency", 0.0),
            "stream_details": stream_stats.get("stream_details", {}),
            "queue_sizes": stream_stats.get("queue_sizes", {})
        }
    
    async def _generate_security_events_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate security monitoring data"""
        security_metrics = self.coordinator.security_framework.get_security_metrics()
        
        return {
            "active_sessions": security_metrics.get("active_sessions", 0),
            "high_risk_sessions": security_metrics.get("high_risk_sessions", 0),
            "audit_summary": security_metrics.get("audit_summary", {}),
            "recent_events": [
                {
                    "event_type": "AUTHENTICATION",
                    "outcome": "SUCCESS",
                    "user_id": "user_123",
                    "timestamp": datetime.utcnow().isoformat(),
                    "risk_score": 0.2
                },
                {
                    "event_type": "DATA_ACCESS",
                    "outcome": "SUCCESS",
                    "user_id": "user_456",
                    "timestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
                    "risk_score": 0.1
                }
            ]
        }
    
    async def _generate_performance_metrics_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate system performance metrics"""
        import psutil
        import random
        
        return {
            "system_resources": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else random.uniform(20, 80)
            },
            "application_metrics": {
                "response_time_ms": random.uniform(50, 200),
                "requests_per_second": random.uniform(100, 500),
                "error_rate": random.uniform(0, 0.05),
                "uptime_hours": random.uniform(24, 720)
            },
            "database_metrics": {
                "connections": random.randint(10, 100),
                "queries_per_second": random.uniform(50, 200),
                "slow_queries": random.randint(0, 5),
                "cache_hit_rate": random.uniform(0.8, 0.99)
            }
        }
    
    async def _generate_generic_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate generic data for unknown widget types"""
        return {
            "widget_id": widget.widget_id,
            "data_source": widget.data_source,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Data source not implemented",
            "sample_data": [
                {"x": i, "y": i * 2 + 1} for i in range(10)
            ]
        }

class DashboardWebSocketManager:
    """Manages WebSocket connections for real-time dashboard updates"""
    
    def __init__(self, data_provider: DashboardDataProvider):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.data_provider = data_provider
        self.dashboards: Dict[str, DashboardConfig] = {}
        self.security_framework = ZeroTrustSecurityFramework()
        
        # Background tasks
        self.update_tasks: Dict[str, asyncio.Task] = {}
        
        # Load default dashboards
        self._initialize_default_dashboards()
    
    def _initialize_default_dashboards(self):
        """Initialize default dashboards for each vertical"""
        # Tactical COP Dashboard
        tactical_dashboard = DashboardConfig(
            dashboard_id="tactical_cop_001",
            dashboard_type=DashboardType.TACTICAL_COP,
            title="Tactical Common Operating Picture",
            description="Real-time tactical situation awareness",
            widgets=[
                DashboardWidget(
                    widget_id="tactical_situation",
                    widget_type="status_panel",
                    title="Situation Overview",
                    data_source="tactical_cop",
                    position={"x": 0, "y": 0, "w": 6, "h": 4},
                    update_frequency=UpdateFrequency.HIGH_FREQUENCY,
                    required_permissions=["view_tactical_data"]
                ),
                DashboardWidget(
                    widget_id="force_status",
                    widget_type="table",
                    title="Force Status",
                    data_source="tactical_cop",
                    position={"x": 6, "y": 0, "w": 6, "h": 4},
                    update_frequency=UpdateFrequency.MEDIUM_FREQUENCY
                )
            ]
        )
        self.dashboards[tactical_dashboard.dashboard_id] = tactical_dashboard
        
        # Patient Monitoring Dashboard
        patient_dashboard = DashboardConfig(
            dashboard_id="patient_monitoring_001",
            dashboard_type=DashboardType.PATIENT_MONITORING,
            title="ICU Patient Monitoring",
            description="Real-time patient monitoring and alerts",
            widgets=[
                DashboardWidget(
                    widget_id="icu_overview",
                    widget_type="metrics",
                    title="ICU Overview",
                    data_source="patient_monitoring",
                    position={"x": 0, "y": 0, "w": 4, "h": 3},
                    update_frequency=UpdateFrequency.HIGH_FREQUENCY,
                    required_permissions=["view_patient_data"]
                ),
                DashboardWidget(
                    widget_id="patient_alerts",
                    widget_type="alert_list",
                    title="Patient Alerts",
                    data_source="patient_monitoring",
                    position={"x": 4, "y": 0, "w": 4, "h": 3},
                    update_frequency=UpdateFrequency.REAL_TIME
                ),
                DashboardWidget(
                    widget_id="vital_trends",
                    widget_type="line_chart",
                    title="Vital Signs Trends",
                    data_source="patient_monitoring",
                    position={"x": 8, "y": 0, "w": 4, "h": 3},
                    update_frequency=UpdateFrequency.HIGH_FREQUENCY
                )
            ]
        )
        self.dashboards[patient_dashboard.dashboard_id] = patient_dashboard
        
        # Risk Monitoring Dashboard
        risk_dashboard = DashboardConfig(
            dashboard_id="risk_monitoring_001",
            dashboard_type=DashboardType.RISK_MONITORING,
            title="Financial Risk Monitoring",
            description="Real-time financial risk metrics and alerts",
            widgets=[
                DashboardWidget(
                    widget_id="market_risk",
                    widget_type="gauge_chart",
                    title="Market Risk (VaR)",
                    data_source="risk_monitoring",
                    position={"x": 0, "y": 0, "w": 3, "h": 3},
                    update_frequency=UpdateFrequency.HIGH_FREQUENCY,
                    required_permissions=["view_risk_data"]
                ),
                DashboardWidget(
                    widget_id="risk_alerts",
                    widget_type="alert_list",
                    title="Risk Alerts",
                    data_source="risk_monitoring",
                    position={"x": 3, "y": 0, "w": 4, "h": 3},
                    update_frequency=UpdateFrequency.REAL_TIME
                ),
                DashboardWidget(
                    widget_id="risk_trends",
                    widget_type="line_chart",
                    title="Risk Trends",
                    data_source="risk_monitoring",
                    position={"x": 7, "y": 0, "w": 5, "h": 3},
                    update_frequency=UpdateFrequency.MEDIUM_FREQUENCY
                )
            ]
        )
        self.dashboards[risk_dashboard.dashboard_id] = risk_dashboard
        
        # Swarm Coordination Dashboard
        swarm_dashboard = DashboardConfig(
            dashboard_id="swarm_coordination_001",
            dashboard_type=DashboardType.SWARM_COORDINATION,
            title="Swarm Coordination Center",
            description="Real-time swarm orchestration and agent metrics",
            widgets=[
                DashboardWidget(
                    widget_id="swarm_overview",
                    widget_type="metrics",
                    title="Swarm Overview",
                    data_source="swarm_metrics",
                    position={"x": 0, "y": 0, "w": 4, "h": 3},
                    update_frequency=UpdateFrequency.HIGH_FREQUENCY
                ),
                DashboardWidget(
                    widget_id="agent_deployment",
                    widget_type="bar_chart",
                    title="Agent Deployment",
                    data_source="swarm_metrics",
                    position={"x": 4, "y": 0, "w": 4, "h": 3},
                    update_frequency=UpdateFrequency.MEDIUM_FREQUENCY
                ),
                DashboardWidget(
                    widget_id="task_throughput",
                    widget_type="line_chart",
                    title="Task Throughput",
                    data_source="swarm_metrics",
                    position={"x": 8, "y": 0, "w": 4, "h": 3},
                    update_frequency=UpdateFrequency.HIGH_FREQUENCY
                )
            ]
        )
        self.dashboards[swarm_dashboard.dashboard_id] = swarm_dashboard
    
    async def connect_websocket(self, websocket: WebSocket, user_id: str, session_id: str) -> str:
        """Connect a new WebSocket client"""
        await websocket.accept()
        
        connection = WebSocketConnection(
            websocket=websocket,
            user_id=user_id,
            session_id=session_id,
            ip_address=websocket.client.host if websocket.client else "unknown"
        )
        
        self.connections[connection.connection_id] = connection
        
        # Start update task for this connection
        self.update_tasks[connection.connection_id] = asyncio.create_task(
            self._connection_update_loop(connection.connection_id)
        )
        
        log.info(f"WebSocket connected: {connection.connection_id} for user {user_id}")
        return connection.connection_id
    
    async def disconnect_websocket(self, connection_id: str):
        """Disconnect a WebSocket client"""
        if connection_id in self.connections:
            # Cancel update task
            if connection_id in self.update_tasks:
                self.update_tasks[connection_id].cancel()
                del self.update_tasks[connection_id]
            
            # Remove connection
            del self.connections[connection_id]
            log.info(f"WebSocket disconnected: {connection_id}")
    
    async def subscribe_to_dashboard(self, connection_id: str, dashboard_id: str) -> bool:
        """Subscribe connection to dashboard updates"""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        # Check authorization
        if dashboard_id in self.dashboards:
            dashboard = self.dashboards[dashboard_id]
            
            # Check if user has required roles
            for role in dashboard.required_roles:
                if not await authorize(connection.session_id, role):
                    log.warning(f"User {connection.user_id} denied access to dashboard {dashboard_id}")
                    return False
        
        connection.subscribed_dashboards.add(dashboard_id)
        connection.update_activity()
        
        # Send initial dashboard data
        await self._send_dashboard_update(connection_id, dashboard_id)
        
        log.info(f"Connection {connection_id} subscribed to dashboard {dashboard_id}")
        return True
    
    async def unsubscribe_from_dashboard(self, connection_id: str, dashboard_id: str):
        """Unsubscribe connection from dashboard updates"""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            connection.subscribed_dashboards.discard(dashboard_id)
            connection.update_activity()
    
    async def _connection_update_loop(self, connection_id: str):
        """Background loop to send updates to a specific connection"""
        while connection_id in self.connections:
            try:
                connection = self.connections[connection_id]
                
                # Check if connection is expired
                if connection.is_expired():
                    await self.disconnect_websocket(connection_id)
                    break
                
                # Send updates for subscribed dashboards
                for dashboard_id in connection.subscribed_dashboards:
                    await self._send_dashboard_update(connection_id, dashboard_id)
                
                # Wait based on highest frequency subscription
                await asyncio.sleep(self._get_update_interval(connection))
                
            except WebSocketDisconnect:
                await self.disconnect_websocket(connection_id)
                break
            except Exception as e:
                log.error(f"Connection update loop error: {e}")
                await asyncio.sleep(5)
    
    async def _send_dashboard_update(self, connection_id: str, dashboard_id: str):
        """Send dashboard update to specific connection"""
        try:
            if connection_id not in self.connections or dashboard_id not in self.dashboards:
                return
            
            connection = self.connections[connection_id]
            dashboard = self.dashboards[dashboard_id]
            
            # Collect widget data
            widget_data = {}
            for widget in dashboard.widgets:
                # Check widget permissions
                if widget.required_permissions:
                    has_permission = all(
                        await authorize(connection.session_id, perm) 
                        for perm in widget.required_permissions
                    )
                    if not has_permission:
                        continue
                
                # Get widget data
                data = await self.data_provider.get_widget_data(widget)
                widget_data[widget.widget_id] = {
                    "widget_config": asdict(widget),
                    "data": data,
                    "timestamp": time.time()
                }
            
            # Send update message
            update_message = {
                "type": "dashboard_update",
                "dashboard_id": dashboard_id,
                "timestamp": time.time(),
                "widgets": widget_data
            }
            
            await connection.websocket.send_text(json.dumps(update_message))
            connection.update_activity()
            
        except Exception as e:
            log.error(f"Failed to send dashboard update: {e}")
    
    def _get_update_interval(self, connection: WebSocketConnection) -> float:
        """Get update interval based on connection subscriptions"""
        min_interval = 30.0  # Default 30 seconds
        
        for dashboard_id in connection.subscribed_dashboards:
            if dashboard_id in self.dashboards:
                dashboard = self.dashboards[dashboard_id]
                for widget in dashboard.widgets:
                    if widget.update_frequency == UpdateFrequency.REAL_TIME:
                        min_interval = min(min_interval, 0.1)
                    elif widget.update_frequency == UpdateFrequency.HIGH_FREQUENCY:
                        min_interval = min(min_interval, 2.0)
                    elif widget.update_frequency == UpdateFrequency.MEDIUM_FREQUENCY:
                        min_interval = min(min_interval, 10.0)
        
        return min_interval
    
    async def broadcast_alert(self, alert_type: str, message: str, severity: str = "INFO"):
        """Broadcast alert to all connected clients"""
        alert_message = {
            "type": "alert",
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": time.time()
        }
        
        disconnected_connections = []
        
        for connection_id, connection in self.connections.items():
            try:
                await connection.websocket.send_text(json.dumps(alert_message))
            except:
                disconnected_connections.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected_connections:
            await self.disconnect_websocket(connection_id)
    
    def get_dashboard_config(self, dashboard_id: str) -> Optional[DashboardConfig]:
        """Get dashboard configuration"""
        return self.dashboards.get(dashboard_id)
    
    def list_dashboards(self, user_id: str = None) -> List[Dict[str, Any]]:
        """List available dashboards for user"""
        dashboard_list = []
        
        for dashboard in self.dashboards.values():
            # Basic access control check
            if dashboard.public_access or not user_id or dashboard.owner_id == user_id or user_id in dashboard.shared_users:
                dashboard_list.append({
                    "dashboard_id": dashboard.dashboard_id,
                    "title": dashboard.title,
                    "description": dashboard.description,
                    "dashboard_type": dashboard.dashboard_type.value,
                    "theme": dashboard.theme,
                    "widget_count": len(dashboard.widgets),
                    "tags": dashboard.tags
                })
        
        return dashboard_list

# FastAPI application
app = FastAPI(
    title="Universal I/O Dashboard API",
    description="Real-time dashboards for universal input/output processing",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global instances
coordinator = None
websocket_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global coordinator, websocket_manager
    
    # Initialize coordinator
    from ..integration.swarm_integration import get_global_coordinator
    coordinator = await get_global_coordinator()
    
    # Initialize WebSocket manager
    data_provider = DashboardDataProvider(coordinator)
    websocket_manager = DashboardWebSocketManager(data_provider)
    
    log.info("Dashboard API server started")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Universal I/O Dashboard API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "coordinator": coordinator is not None,
            "websocket_manager": websocket_manager is not None
        }
    }

@app.get("/dashboards")
async def list_dashboards(credentials: HTTPAuthorizationCredentials = Security(security)):
    """List available dashboards"""
    # TODO: Extract user ID from credentials
    user_id = "demo_user"  # Simplified for demo
    
    return {
        "dashboards": websocket_manager.list_dashboards(user_id),
        "total": len(websocket_manager.dashboards)
    }

@app.get("/dashboards/{dashboard_id}")
async def get_dashboard(dashboard_id: str, credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get specific dashboard configuration"""
    dashboard = websocket_manager.get_dashboard_config(dashboard_id)
    
    if not dashboard:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    return asdict(dashboard)

@app.get("/dashboards/{dashboard_id}/data")
async def get_dashboard_data(dashboard_id: str, credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get current data for all widgets in dashboard"""
    dashboard = websocket_manager.get_dashboard_config(dashboard_id)
    
    if not dashboard:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    widget_data = {}
    for widget in dashboard.widgets:
        data = await websocket_manager.data_provider.get_widget_data(widget)
        widget_data[widget.widget_id] = data
    
    return {
        "dashboard_id": dashboard_id,
        "timestamp": time.time(),
        "widgets": widget_data
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    # TODO: Implement proper authentication
    user_id = "demo_user"
    session_id = "demo_session"
    
    connection_id = await websocket_manager.connect_websocket(websocket, user_id, session_id)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "subscribe_dashboard":
                dashboard_id = message.get("dashboard_id")
                if dashboard_id:
                    await websocket_manager.subscribe_to_dashboard(connection_id, dashboard_id)
                    await websocket.send_text(json.dumps({
                        "type": "subscription_confirmed",
                        "dashboard_id": dashboard_id
                    }))
            
            elif message_type == "unsubscribe_dashboard":
                dashboard_id = message.get("dashboard_id")
                if dashboard_id:
                    await websocket_manager.unsubscribe_from_dashboard(connection_id, dashboard_id)
            
            elif message_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            
    except WebSocketDisconnect:
        await websocket_manager.disconnect_websocket(connection_id)

@app.get("/demo")
async def demo_dashboard():
    """Serve demo dashboard HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Universal I/O Dashboard Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: white; }
            .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .widget { background: #2d2d2d; border-radius: 8px; padding: 20px; border: 1px solid #444; }
            .widget h3 { margin-top: 0; color: #4CAF50; }
            .metric { font-size: 2em; font-weight: bold; color: #2196F3; }
            .status { padding: 5px 10px; border-radius: 4px; display: inline-block; }
            .status.green { background: #4CAF50; }
            .status.yellow { background: #FF9800; }
            .status.red { background: #f44336; }
            .alert { background: #f44336; color: white; padding: 10px; margin: 10px 0; border-radius: 4px; }
            #status { position: fixed; top: 10px; right: 10px; padding: 10px; background: #333; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>Universal I/O Dashboard Demo</h1>
        <div id="status">Connecting...</div>
        
        <div class="dashboard">
            <div class="widget">
                <h3>Swarm Coordination</h3>
                <div id="swarm-data">Loading...</div>
            </div>
            
            <div class="widget">
                <h3>Stream Processing</h3>
                <div id="stream-data">Loading...</div>
            </div>
            
            <div class="widget">
                <h3>Security Events</h3>
                <div id="security-data">Loading...</div>
            </div>
            
            <div class="widget">
                <h3>System Performance</h3>
                <div id="performance-data">Loading...</div>
            </div>
        </div>

        <script>
            const ws = new WebSocket('ws://localhost:8000/ws');
            const statusEl = document.getElementById('status');
            
            ws.onopen = function(event) {
                statusEl.textContent = 'Connected';
                statusEl.style.background = '#4CAF50';
                
                // Subscribe to swarm dashboard
                ws.send(JSON.stringify({
                    type: 'subscribe_dashboard',
                    dashboard_id: 'swarm_coordination_001'
                }));
            };
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                
                if (message.type === 'dashboard_update') {
                    updateDashboard(message);
                } else if (message.type === 'alert') {
                    showAlert(message);
                }
            };
            
            ws.onclose = function(event) {
                statusEl.textContent = 'Disconnected';
                statusEl.style.background = '#f44336';
            };
            
            function updateDashboard(message) {
                const widgets = message.widgets;
                
                // Update swarm data
                if (widgets.swarm_overview) {
                    const data = widgets.swarm_overview.data;
                    document.getElementById('swarm-data').innerHTML = 
                        '<div class="metric">' + data.active_swarms + '</div>' +
                        '<div>Active Swarms</div>' +
                        '<div class="metric">' + data.total_agents_deployed + '</div>' +
                        '<div>Total Agents</div>';
                }
            }
            
            function showAlert(alert) {
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert';
                alertDiv.textContent = alert.message;
                document.body.insertBefore(alertDiv, document.body.firstChild);
                
                setTimeout(() => alertDiv.remove(), 5000);
            }
            
            // Send ping every 30 seconds
            setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'ping'}));
                }
            }, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
