"""
Universal I/O Service - Main Entry Point
Comprehensive universal input/output processing service with:
- Real-time stream ingestion (Kafka, WebSocket, REST, file watchers)
- Event processing pipeline (Flink-style complex event processing)  
- Specialized vertical outputs (defense, healthcare, finance, BI, federal)
- Zero-trust security framework with encryption and audit logging
- Swarm orchestration integration for massive scale processing (400+ agents)
- Real-time dashboard APIs with WebSocket streaming
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('universal-io.log')
    ]
)

log = logging.getLogger("universal-io-service")

# Import service components
from .stream.stream_ingestion import StreamIngestionEngine, create_file_watcher, create_market_data_stream
from .stream.event_processor import EventProcessingEngine, create_financial_processing_pipeline
from .outputs.vertical_generators import VerticalDomain
from .security.zero_trust_framework import ZeroTrustSecurityFramework, SecurityLevel
from .integration.swarm_integration import UniversalIOSwarmCoordinator, ProcessingScale, ProcessingObjective, SwarmProcessingTask, CodebaseAnalysisTask
from .integration.legacy_integration import UniversalIOIntegrationLayer, get_integration_layer
from .api.dashboard_server import app as dashboard_app

@dataclass
class ServiceConfig:
    """Configuration for Universal I/O Service"""
    # Service settings
    service_name: str = "universal-io"
    service_version: str = "1.0.0"
    environment: str = "development"  # development, staging, production
    
    # Stream processing
    enable_stream_ingestion: bool = True
    enable_event_processing: bool = True
    max_concurrent_streams: int = 1000
    
    # Swarm integration
    enable_swarm_integration: bool = True
    default_processing_scale: ProcessingScale = ProcessingScale.MEDIUM_SWARM
    
    # Security
    enable_security_framework: bool = True
    default_security_level: SecurityLevel = SecurityLevel.INTERNAL
    enable_audit_logging: bool = True
    
    # Dashboard API
    enable_dashboard_api: bool = True
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000
    
    # Performance
    worker_threads: int = 20
    max_memory_usage_gb: int = 16
    enable_metrics: bool = True
    
    # File paths
    config_dir: str = "config"
    data_dir: str = "data"
    logs_dir: str = "logs"
    
    @classmethod
    def from_env(cls) -> 'ServiceConfig':
        """Load configuration from environment variables"""
        return cls(
            environment=os.getenv("UNIVERSAL_IO_ENV", "development"),
            enable_swarm_integration=os.getenv("ENABLE_SWARM", "true").lower() == "true",
            enable_security_framework=os.getenv("ENABLE_SECURITY", "true").lower() == "true",
            enable_dashboard_api=os.getenv("ENABLE_DASHBOARD", "true").lower() == "true",
            dashboard_host=os.getenv("DASHBOARD_HOST", "0.0.0.0"),
            dashboard_port=int(os.getenv("DASHBOARD_PORT", "8000")),
            max_memory_usage_gb=int(os.getenv("MAX_MEMORY_GB", "16")),
            worker_threads=int(os.getenv("WORKER_THREADS", "20"))
        )

class UniversalIOService:
    """Main Universal I/O Service"""
    
    def __init__(self, config: ServiceConfig = None):
        self.config = config or ServiceConfig.from_env()
        
        # Core components
        self.stream_engine: Optional[StreamIngestionEngine] = None
        self.event_processor: Optional[EventProcessingEngine] = None
        self.swarm_coordinator: Optional[UniversalIOSwarmCoordinator] = None
        self.security_framework: Optional[ZeroTrustSecurityFramework] = None
        
        # Integration layer (combines legacy and new capabilities)
        self.integration_layer: Optional[UniversalIOIntegrationLayer] = None
        
        # Service state
        self.is_running = False
        self.start_time = 0.0
        self.shutdown_event = asyncio.Event()
        
        # Performance metrics
        self.metrics = {
            "service_uptime": 0.0,
            "total_requests_processed": 0,
            "streams_active": 0,
            "agents_deployed": 0,
            "security_events": 0,
            "dashboard_connections": 0,
            "memory_usage_mb": 0.0,
            "cpu_usage_percent": 0.0
        }
        
        log.info(f"Universal I/O Service initialized - Environment: {self.config.environment}")
    
    async def initialize(self) -> bool:
        """Initialize all service components"""
        try:
            log.info("Initializing Universal I/O Service...")
            
            # Create directories
            self._create_directories()
            
            # Initialize stream ingestion engine
            if self.config.enable_stream_ingestion:
                log.info("Initializing stream ingestion engine...")
                self.stream_engine = StreamIngestionEngine(
                    max_concurrent_streams=self.config.max_concurrent_streams
                )
            
            # Initialize event processing engine
            if self.config.enable_event_processing:
                log.info("Initializing event processing engine...")
                self.event_processor = EventProcessingEngine()
                await self.event_processor.start_processing()
            
            # Initialize security framework
            if self.config.enable_security_framework:
                log.info("Initializing security framework...")
                self.security_framework = ZeroTrustSecurityFramework()
            
            # Initialize swarm coordinator
            if self.config.enable_swarm_integration:
                log.info("Initializing swarm coordinator...")
                self.swarm_coordinator = UniversalIOSwarmCoordinator()
                await self.swarm_coordinator.initialize()
            
            # Initialize integration layer (combines all capabilities)
            log.info("Initializing integration layer...")
            self.integration_layer = UniversalIOIntegrationLayer()
            await self.integration_layer.initialize(
                stream_engine=self.stream_engine,
                event_processor=self.event_processor,
                security_framework=self.security_framework
            )
            
            # Set up default stream processing pipelines
            await self._setup_default_pipelines()
            
            # Start background monitoring
            asyncio.create_task(self._monitoring_loop())
            
            log.info("Universal I/O Service initialization complete")
            return True
            
        except Exception as e:
            log.error(f"Service initialization failed: {e}")
            return False
    
    def _create_directories(self):
        """Create necessary directories"""
        for dir_name in [self.config.config_dir, self.config.data_dir, self.config.logs_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    async def _setup_default_pipelines(self):
        """Set up default processing pipelines"""
        if not self.event_processor:
            return
        
        try:
            # Set up financial processing pipeline
            financial_pipeline = create_financial_processing_pipeline()
            
            # Add operators to event processor
            for _, operator in financial_pipeline.operators.items():
                self.event_processor.add_operator(operator)
            
            # Connect operators based on topology
            for source_id, targets in financial_pipeline.processing_topology.items():
                for target_id in targets:
                    self.event_processor.connect_operators(source_id, target_id)
            
            log.info("Default processing pipelines configured")
            
        except Exception as e:
            log.error(f"Failed to setup default pipelines: {e}")
    
    async def start(self) -> bool:
        """Start the Universal I/O Service"""
        try:
            if self.is_running:
                log.warning("Service is already running")
                return True
            
            log.info("Starting Universal I/O Service...")
            
            # Initialize if not already done
            if not await self.initialize():
                return False
            
            self.is_running = True
            self.start_time = time.time()
            
            # Start demo streams for testing
            await self._start_demo_streams()
            
            # Start dashboard API if enabled
            if self.config.enable_dashboard_api:
                asyncio.create_task(self._start_dashboard_api())
            
            log.info(f"Universal I/O Service started successfully on {self.config.environment} environment")
            return True
            
        except Exception as e:
            log.error(f"Failed to start service: {e}")
            self.is_running = False
            return False
    
    async def _start_demo_streams(self):
        """Start demo streams for testing"""
        if not self.stream_engine:
            return
        
        try:
            # Market data stream
            market_config = await create_market_data_stream("demo_market_data")
            await self.stream_engine.start_stream(market_config)
            
            # File watcher for a demo directory
            demo_dir = Path(self.config.data_dir) / "demo"
            demo_dir.mkdir(exist_ok=True)
            
            file_config = await create_file_watcher(str(demo_dir), "demo_file_watcher")
            await self.stream_engine.start_stream(file_config)
            
            log.info("Demo streams started")
            
        except Exception as e:
            log.error(f"Failed to start demo streams: {e}")
    
    async def _start_dashboard_api(self):
        """Start the dashboard API server"""
        try:
            import uvicorn
            config = uvicorn.Config(
                dashboard_app,
                host=self.config.dashboard_host,
                port=self.config.dashboard_port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            
            log.info(f"Starting dashboard API on http://{self.config.dashboard_host}:{self.config.dashboard_port}")
            await server.serve()
            
        except Exception as e:
            log.error(f"Dashboard API failed: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_running:
            try:
                await self._update_metrics()
                await self._check_health()
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                log.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    async def _update_metrics(self):
        """Update service metrics"""
        try:
            self.metrics["service_uptime"] = time.time() - self.start_time
            
            # Stream metrics
            if self.stream_engine:
                stream_stats = self.stream_engine.get_stream_stats()
                self.metrics["streams_active"] = stream_stats["global_stats"]["active_streams"]
            
            # Swarm metrics
            if self.swarm_coordinator:
                swarm_metrics = self.swarm_coordinator.get_coordinator_metrics()
                self.metrics["agents_deployed"] = swarm_metrics["processing_metrics"].get("peak_agents_deployed", 0)
            
            # Security metrics
            if self.security_framework:
                security_metrics = self.security_framework.get_security_metrics()
                self.metrics["security_events"] = security_metrics["audit_summary"].get("total_events", 0)
            
            # System metrics
            try:
                import psutil
                self.metrics["memory_usage_mb"] = psutil.virtual_memory().used / (1024 * 1024)
                self.metrics["cpu_usage_percent"] = psutil.cpu_percent()
            except ImportError:
                pass
            
        except Exception as e:
            log.error(f"Failed to update metrics: {e}")
    
    async def _check_health(self):
        """Check service health"""
        try:
            issues = []
            
            # Check memory usage
            if self.metrics["memory_usage_mb"] > self.config.max_memory_usage_gb * 1024:
                issues.append(f"High memory usage: {self.metrics['memory_usage_mb']:.1f} MB")
            
            # Check CPU usage
            if self.metrics["cpu_usage_percent"] > 90:
                issues.append(f"High CPU usage: {self.metrics['cpu_usage_percent']:.1f}%")
            
            # Log health issues
            if issues:
                log.warning(f"Health check issues: {', '.join(issues)}")
            
        except Exception as e:
            log.error(f"Health check failed: {e}")
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a universal I/O request using integrated pipeline"""
        try:
            request_id = request_data.get("request_id", f"req_{int(time.time())}")
            
            # Extract request parameters
            input_data = request_data.get("input_data")
            output_format = request_data.get("output_format", "json")
            vertical_domain = request_data.get("vertical_domain")
            processing_scale = request_data.get("processing_scale", "medium_swarm")
            security_level = request_data.get("security_level", "internal")
            use_swarm = request_data.get("use_swarm", True)
            
            log.info(f"Processing request {request_id}")
            
            # Process through integration layer (automatically routes to best system)
            if self.integration_layer:
                result = await self.integration_layer.process_universal_request(
                    input_data=input_data,
                    output_format=output_format,
                    vertical_domain=vertical_domain,
                    use_advanced_processors=True,
                    use_stream_processing=request_data.get("use_stream_processing", False),
                    security_level=security_level,
                    requirements=request_data.get("requirements", {}),
                    metadata=request_data.get("metadata", {}),
                    quality=request_data.get("quality", "production")
                )
                
                # Add request tracking
                result["request_id"] = request_id
                result["status"] = "completed" if result.get("success", False) else "failed"
                
                self.metrics["total_requests_processed"] += 1
                return result
            
            # Fallback to swarm coordinator if integration layer not available
            elif self.swarm_coordinator and use_swarm:
                task = SwarmProcessingTask(
                    input_data=input_data,
                    output_format=output_format,
                    vertical_domain=VerticalDomain(vertical_domain) if vertical_domain else None,
                    scale=ProcessingScale(processing_scale),
                    security_level=SecurityLevel(security_level),
                    objective=ProcessingObjective.COMPREHENSIVE_ANALYSIS
                )
                
                task_id = await self.swarm_coordinator.submit_processing_task(task)
                
                # Wait for completion (simplified - in production, use async polling)
                max_wait = 300  # 5 minutes
                wait_interval = 1
                waited = 0
                
                while waited < max_wait:
                    status = await self.swarm_coordinator.get_task_status(task_id)
                    if status and status.get("status") == "completed":
                        self.metrics["total_requests_processed"] += 1
                        return {
                            "request_id": request_id,
                            "status": "completed",
                            "result": status.get("result"),
                            "confidence": status.get("confidence", 0.0),
                            "agents_used": status.get("agents_used", 0),
                            "processing_time": status.get("processing_time", 0.0),
                            "processing_method": "swarm_coordinator"
                        }
                    
                    await asyncio.sleep(wait_interval)
                    waited += wait_interval
                
                return {
                    "request_id": request_id,
                    "status": "timeout",
                    "error": "Processing timeout"
                }
            else:
                # Basic fallback processing
                return {
                    "request_id": request_id,
                    "status": "completed",
                    "result": {"processed_data": input_data, "method": "basic_fallback"},
                    "confidence": 0.5,
                    "agents_used": 1,
                    "processing_time": 0.1,
                    "processing_method": "fallback"
                }
                
        except Exception as e:
            log.error(f"Request processing failed: {e}")
            return {
                "request_id": request_data.get("request_id", "unknown"),
                "status": "failed",
                "error": str(e),
                "processing_method": "error"
            }
    
    async def analyze_codebase(self, codebase_path: str, analysis_depth: str = "comprehensive") -> Dict[str, Any]:
        """Analyze entire codebase with specialized agents"""
        try:
            if not self.swarm_coordinator:
                raise ValueError("Swarm coordinator not available")
            
            log.info(f"Starting codebase analysis: {codebase_path}")
            
            # Create analysis task
            analysis_task = CodebaseAnalysisTask(
                codebase_path=codebase_path,
                analysis_depth=analysis_depth,
                discover_capabilities=True,
                map_integrations=True,
                assess_quality=True,
                security_analysis=True,
                performance_analysis=True
            )
            
            # Submit to swarm coordinator
            task_id = await self.swarm_coordinator.submit_codebase_analysis(analysis_task)
            
            log.info(f"Codebase analysis submitted: {task_id}")
            
            return {
                "analysis_id": analysis_task.analysis_id,
                "task_id": task_id,
                "status": "submitted",
                "codebase_path": codebase_path,
                "analysis_depth": analysis_depth,
                "estimated_agents": sum(analysis_task.specialized_agents.values())
            }
            
        except Exception as e:
            log.error(f"Codebase analysis failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        return {
            "service": {
                "name": self.config.service_name,
                "version": self.config.service_version,
                "environment": self.config.environment,
                "is_running": self.is_running,
                "uptime_seconds": self.metrics["service_uptime"]
            },
            "components": {
                "stream_engine": self.stream_engine is not None,
                "event_processor": self.event_processor is not None,
                "swarm_coordinator": self.swarm_coordinator is not None,
                "security_framework": self.security_framework is not None
            },
            "metrics": self.metrics,
            "configuration": {
                "max_concurrent_streams": self.config.max_concurrent_streams,
                "default_processing_scale": self.config.default_processing_scale.value,
                "dashboard_enabled": self.config.enable_dashboard_api,
                "dashboard_url": f"http://{self.config.dashboard_host}:{self.config.dashboard_port}" if self.config.enable_dashboard_api else None
            }
        }
    
    async def shutdown(self):
        """Shutdown the service gracefully"""
        log.info("Shutting down Universal I/O Service...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        try:
            # Shutdown components in reverse order
            if self.swarm_coordinator:
                await self.swarm_coordinator.shutdown()
            
            if self.event_processor:
                await self.event_processor.shutdown()
            
            if self.stream_engine:
                await self.stream_engine.shutdown()
            
            log.info("Universal I/O Service shutdown complete")
            
        except Exception as e:
            log.error(f"Error during shutdown: {e}")

# Global service instance
_service_instance: Optional[UniversalIOService] = None

async def get_service() -> UniversalIOService:
    """Get global service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = UniversalIOService()
        await _service_instance.start()
    return _service_instance

# CLI interface
async def main():
    """Main entry point for CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal I/O Service")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--env", default="development", help="Environment (development, staging, production)")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard API port")
    parser.add_argument("--analyze-codebase", help="Path to codebase for analysis")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    
    args = parser.parse_args()
    
    # Create service configuration
    config = ServiceConfig.from_env()
    config.environment = args.env
    config.dashboard_port = args.port
    
    # Create and start service
    service = UniversalIOService(config)
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        log.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(service.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start service
    if not await service.start():
        log.error("Failed to start service")
        return 1
    
    # Handle specific operations
    if args.analyze_codebase:
        result = await service.analyze_codebase(args.analyze_codebase)
        print(json.dumps(result, indent=2))
        return 0
    
    # Print service status
    status = service.get_service_status()
    print(json.dumps(status, indent=2))
    
    if args.demo:
        print(f"\nDemo mode - Dashboard available at: http://localhost:{config.dashboard_port}/demo")
        print("WebSocket endpoint: ws://localhost:{}/ws".format(config.dashboard_port))
        print("Press Ctrl+C to shutdown")
    
    # Keep service running
    try:
        await service.shutdown_event.wait()
    except KeyboardInterrupt:
        pass
    
    await service.shutdown()
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log.info("Service interrupted by user")
        sys.exit(0)
    except Exception as e:
        log.error(f"Service failed: {e}")
        sys.exit(1)
