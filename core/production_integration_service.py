"""
Production Integration Service
Coordinates all external services, databases, and infrastructure components
"""
import asyncio
import logging
import os
from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass

# Import all production managers
from .database_manager import get_database_manager, DatabaseManager
from .cache_manager import get_cache_manager, CacheManager
from .external_services_manager import get_external_services_manager, ExternalServicesManager

log = logging.getLogger("production-integration")

@dataclass
class IntegrationStatus:
    """Status of all integrated services"""
    database: bool = False
    cache: bool = False
    external_services: bool = False
    pinecone: bool = False
    neo4j: bool = False
    kafka: bool = False
    monitoring: bool = False

class ProductionIntegrationService:
    """Coordinates all production infrastructure and services"""
    
    def __init__(self):
        self.database_manager: Optional[DatabaseManager] = None
        self.cache_manager: Optional[CacheManager] = None
        self.external_services: Optional[ExternalServicesManager] = None
        
        self._initialized = False
        self.status = IntegrationStatus()
        
        # Performance metrics
        self.metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "database_query_time": 0.0,
            "external_api_calls": 0
        }
    
    async def initialize(self):
        """Initialize all production services"""
        if self._initialized:
            return
        
        log.info("Initializing production integration service...")
        
        try:
            # Initialize core infrastructure
            await self._initialize_database()
            await self._initialize_cache()
            await self._initialize_external_services()
            
            # Initialize specialized services
            await self._initialize_pinecone()
            await self._initialize_neo4j()
            await self._initialize_kafka()
            await self._initialize_monitoring()
            
            # Start background tasks
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._metrics_collector())
            
            self._initialized = True
            log.info("Production integration service initialized successfully")
            
        except Exception as e:
            log.error(f"Failed to initialize production integration service: {e}")
            raise
    
    async def _initialize_database(self):
        """Initialize PostgreSQL database manager"""
        try:
            self.database_manager = await get_database_manager()
            self.status.database = True
            log.info("✅ Database manager initialized")
        except Exception as e:
            log.error(f"❌ Database initialization failed: {e}")
            self.status.database = False
    
    async def _initialize_cache(self):
        """Initialize Redis cache manager"""
        try:
            self.cache_manager = await get_cache_manager()
            self.status.cache = True
            log.info("✅ Cache manager initialized")
        except Exception as e:
            log.error(f"❌ Cache initialization failed: {e}")
            self.status.cache = False
    
    async def _initialize_external_services(self):
        """Initialize external services manager"""
        try:
            self.external_services = await get_external_services_manager()
            self.status.external_services = True
            log.info("✅ External services initialized")
        except Exception as e:
            log.error(f"❌ External services initialization failed: {e}")
            self.status.external_services = False
    
    async def _initialize_pinecone(self):
        """Initialize Pinecone vector database"""
        try:
            if os.getenv("PINECONE_API_KEY"):
                import pinecone
                
                pinecone.init(
                    api_key=os.getenv("PINECONE_API_KEY"),
                    environment=os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
                )
                
                # Verify index exists
                index_name = os.getenv("PINECONE_INDEX", "agentforge-production")
                if index_name in pinecone.list_indexes():
                    self.status.pinecone = True
                    log.info("✅ Pinecone vector database connected")
                else:
                    log.warning("⚠️ Pinecone index not found")
            else:
                log.warning("⚠️ Pinecone API key not configured")
        except Exception as e:
            log.error(f"❌ Pinecone initialization failed: {e}")
    
    async def _initialize_neo4j(self):
        """Initialize Neo4j knowledge graph"""
        try:
            if os.getenv("NEO4J_URI"):
                from neo4j import AsyncGraphDatabase
                
                driver = AsyncGraphDatabase.driver(
                    os.getenv("NEO4J_URI"),
                    auth=(
                        os.getenv("NEO4J_USERNAME", "neo4j"),
                        os.getenv("NEO4J_PASSWORD")
                    )
                )
                
                # Test connection
                async with driver.session() as session:
                    result = await session.run("RETURN 1 as test")
                    await result.single()
                
                self.status.neo4j = True
                log.info("✅ Neo4j knowledge graph connected")
                
            else:
                log.warning("⚠️ Neo4j connection not configured")
        except Exception as e:
            log.error(f"❌ Neo4j initialization failed: {e}")
    
    async def _initialize_kafka(self):
        """Initialize Kafka messaging"""
        try:
            if os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
                from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
                
                # Test Kafka connection
                producer = AIOKafkaProducer(
                    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
                    security_protocol=os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
                    sasl_mechanism=os.getenv("KAFKA_SASL_MECHANISM", "PLAIN"),
                    sasl_plain_username=os.getenv("KAFKA_API_KEY"),
                    sasl_plain_password=os.getenv("KAFKA_API_SECRET")
                )
                
                await producer.start()
                await producer.stop()
                
                self.status.kafka = True
                log.info("✅ Kafka messaging connected")
                
            else:
                log.warning("⚠️ Kafka connection not configured")
        except Exception as e:
            log.error(f"❌ Kafka initialization failed: {e}")
    
    async def _initialize_monitoring(self):
        """Initialize monitoring and observability"""
        try:
            # Check if Prometheus endpoint is accessible
            prometheus_endpoint = os.getenv("PROMETHEUS_ENDPOINT")
            jaeger_endpoint = os.getenv("JAEGER_COLLECTOR_ENDPOINT")
            
            if prometheus_endpoint or jaeger_endpoint:
                # Initialize OpenTelemetry if available
                try:
                    from opentelemetry import trace
                    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
                    from opentelemetry.sdk.trace import TracerProvider
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor
                    
                    if jaeger_endpoint:
                        trace.set_tracer_provider(TracerProvider())
                        jaeger_exporter = JaegerExporter(
                            agent_host_name="jaeger-agent",
                            agent_port=6831,
                        )
                        span_processor = BatchSpanProcessor(jaeger_exporter)
                        trace.get_tracer_provider().add_span_processor(span_processor)
                    
                    self.status.monitoring = True
                    log.info("✅ Monitoring and tracing initialized")
                    
                except ImportError:
                    log.warning("⚠️ OpenTelemetry not available")
            else:
                log.warning("⚠️ Monitoring endpoints not configured")
        except Exception as e:
            log.error(f"❌ Monitoring initialization failed: {e}")
    
    async def process_agent_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent request using all available services"""
        start_time = time.time()
        operation_id = f"req_{int(time.time())}_{hash(str(request_data)) % 10000}"
        
        try:
            # Extract request details
            content = request_data.get('content', '')
            capabilities = request_data.get('capabilities', [])
            context = request_data.get('context', {})
            
            # Check cache first
            cached_result = None
            if self.cache_manager:
                cache_key = f"request_{hash(content + str(capabilities))}"
                cached_result = await self.cache_manager.get("agent", cache_key)
            
            if cached_result:
                log.info(f"Cache hit for request {operation_id}")
                self.metrics["cache_hit_rate"] = (self.metrics["cache_hit_rate"] + 1.0) / 2
                return cached_result
            
            # Process with external services
            enriched_context = await self._enrich_context(content, context)
            
            # Store agent execution in database
            if self.database_manager:
                await self.database_manager.store_agent_execution(
                    agent_id=operation_id,
                    swarm_id=context.get('swarm_id', 'default'),
                    task_id=context.get('task_id', operation_id),
                    status='processing',
                    metadata={'capabilities': capabilities, 'enriched_context': enriched_context}
                )
            
            # Generate response using enriched context
            response = await self._generate_enhanced_response(content, capabilities, enriched_context)
            
            # Cache the response
            if self.cache_manager:
                await self.cache_manager.set("agent", cache_key, response, ttl=3600)
            
            # Update database with completion
            if self.database_manager:
                await self.database_manager.store_agent_execution(
                    agent_id=operation_id,
                    swarm_id=context.get('swarm_id', 'default'),
                    task_id=context.get('task_id', operation_id),
                    status='completed',
                    result=response
                )
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(True, processing_time)
            
            return response
            
        except Exception as e:
            log.error(f"Error processing agent request {operation_id}: {e}")
            processing_time = time.time() - start_time
            self._update_metrics(False, processing_time)
            
            # Return error response
            return {
                'error': str(e),
                'operation_id': operation_id,
                'processing_time': processing_time
            }
    
    async def _enrich_context(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich request context with external data sources"""
        enriched = context.copy()
        
        if not self.external_services:
            return enriched
        
        try:
            # Add real-time news if relevant
            if any(keyword in content.lower() for keyword in ['news', 'current', 'recent', 'today']):
                try:
                    news_data = await self.external_services.newsapi_headlines(query=content[:100])
                    enriched['news_context'] = news_data.get('articles', [])[:3]
                except Exception as e:
                    log.warning(f"Failed to fetch news context: {e}")
            
            # Add financial data if relevant
            if any(keyword in content.lower() for keyword in ['stock', 'market', 'finance', 'economy']):
                try:
                    # Extract potential stock symbols
                    words = content.upper().split()
                    stock_symbols = [word for word in words if len(word) <= 5 and word.isalpha()]
                    
                    if stock_symbols:
                        stock_data = await self.external_services.alpha_vantage_stock_quote(stock_symbols[0])
                        enriched['financial_context'] = stock_data
                except Exception as e:
                    log.warning(f"Failed to fetch financial context: {e}")
            
            # Add computational context if relevant
            if any(keyword in content.lower() for keyword in ['calculate', 'compute', 'math', 'formula']):
                try:
                    wolfram_result = await self.external_services.wolfram_query(content)
                    enriched['computational_context'] = wolfram_result
                except Exception as e:
                    log.warning(f"Failed to fetch computational context: {e}")
            
            # Add geospatial context if relevant
            if any(keyword in content.lower() for keyword in ['location', 'address', 'map', 'directions']):
                try:
                    geocoding_result = await self.external_services.mapbox_geocoding(content)
                    enriched['geospatial_context'] = geocoding_result
                except Exception as e:
                    log.warning(f"Failed to fetch geospatial context: {e}")
            
        except Exception as e:
            log.error(f"Error enriching context: {e}")
        
        return enriched
    
    async def _generate_enhanced_response(self, content: str, capabilities: List[str], 
                                        enriched_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced response using all available context"""
        
        # Base response structure
        response = {
            'content': f"Processing request: {content}",
            'capabilities_used': capabilities,
            'context_enriched': bool(enriched_context),
            'data_sources': [],
            'confidence': 0.8,
            'processing_time': time.time()
        }
        
        # Add context-specific enhancements
        if 'news_context' in enriched_context:
            response['data_sources'].append('real-time-news')
            response['content'] += f"\n\nLatest news context: {len(enriched_context['news_context'])} relevant articles found."
        
        if 'financial_context' in enriched_context:
            response['data_sources'].append('financial-markets')
            response['content'] += f"\n\nFinancial data integrated from market sources."
        
        if 'computational_context' in enriched_context:
            response['data_sources'].append('wolfram-alpha')
            response['content'] += f"\n\nComputational analysis performed."
        
        if 'geospatial_context' in enriched_context:
            response['data_sources'].append('geospatial-intelligence')
            response['content'] += f"\n\nGeospatial intelligence integrated."
        
        # Increase confidence based on enriched context
        context_bonus = len(response['data_sources']) * 0.05
        response['confidence'] = min(0.95, response['confidence'] + context_bonus)
        
        return response
    
    def _update_metrics(self, success: bool, processing_time: float):
        """Update performance metrics"""
        self.metrics["total_operations"] += 1
        
        if success:
            self.metrics["successful_operations"] += 1
        else:
            self.metrics["failed_operations"] += 1
        
        # Update average response time
        total_ops = self.metrics["total_operations"]
        self.metrics["avg_response_time"] = (
            (self.metrics["avg_response_time"] * (total_ops - 1) + processing_time) / total_ops
        )
    
    async def _health_monitor(self):
        """Monitor health of all integrated services"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Check database health
                if self.database_manager:
                    try:
                        await self.database_manager.execute_query("SELECT 1", read_only=True)
                        self.status.database = True
                    except Exception:
                        self.status.database = False
                        log.warning("Database health check failed")
                
                # Check cache health
                if self.cache_manager:
                    try:
                        await self.cache_manager.set("health", "check", "ok", ttl=60)
                        self.status.cache = True
                    except Exception:
                        self.status.cache = False
                        log.warning("Cache health check failed")
                
                # Log overall health status
                healthy_services = sum([
                    self.status.database,
                    self.status.cache,
                    self.status.external_services,
                    self.status.pinecone,
                    self.status.neo4j,
                    self.status.kafka,
                    self.status.monitoring
                ])
                
                log.info(f"Health check: {healthy_services}/7 services healthy")
                
            except Exception as e:
                log.error(f"Health monitoring error: {e}")
    
    async def _metrics_collector(self):
        """Collect and aggregate metrics from all services"""
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute
                
                # Collect database metrics
                if self.database_manager:
                    db_stats = await self.database_manager.get_stats()
                    self.metrics["database_query_time"] = db_stats.get("avg_query_time", 0.0)
                
                # Collect cache metrics
                if self.cache_manager:
                    cache_stats = await self.cache_manager.get_stats()
                    self.metrics["cache_hit_rate"] = cache_stats.get("cache_hit_rate", 0.0)
                
                # Collect external services metrics
                if self.external_services:
                    ext_stats = await self.external_services.get_stats()
                    self.metrics["external_api_calls"] = ext_stats.get("total_requests", 0)
                
                log.debug(f"Metrics collected: {self.metrics}")
                
            except Exception as e:
                log.error(f"Metrics collection error: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "status": self.status,
            "metrics": self.metrics,
            "initialized": self._initialized,
            "services": {
                "database": bool(self.database_manager),
                "cache": bool(self.cache_manager),
                "external_services": bool(self.external_services)
            }
        }
    
    async def cleanup(self):
        """Cleanup all services"""
        if self.database_manager:
            await self.database_manager.cleanup()
        
        if self.cache_manager:
            await self.cache_manager.cleanup()
        
        if self.external_services:
            await self.external_services.cleanup()
        
        log.info("Production integration service cleaned up")

# Global production integration service instance
production_integration_service = ProductionIntegrationService()

async def get_production_integration_service() -> ProductionIntegrationService:
    """Get initialized production integration service"""
    if not production_integration_service._initialized:
        await production_integration_service.initialize()
    return production_integration_service
