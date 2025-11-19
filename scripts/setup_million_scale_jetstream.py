#!/usr/bin/env python3
"""
Million-Scale JetStream Setup Script
Configures NATS JetStream for million-agent coordination with:
- High-availability streams with 3+ replicas
- Optimized consumer configurations
- Performance monitoring setup
- Disaster recovery preparation
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import nats
from nats.aio.client import Client as NATS
from nats.js.api import (
    StreamConfig,
    ConsumerConfig,
    RetentionPolicy,
    StorageType,
    DeliverPolicy,
    AckPolicy,
    DiscardPolicy,
    ReplayPolicy,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("jetstream-setup")

# Configuration
NATS_URLS = os.getenv("NATS_URLS", "nats://nats-1:4222,nats://nats-2:4222,nats://nats-3:4222").split(",")
CLUSTER_NAME = os.getenv("NATS_CLUSTER_NAME", "agentforge-million")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

# Million-scale stream configurations
MILLION_SCALE_STREAMS = {
    "agentforge_jobs_million": {
        "description": "Million-scale job distribution with work-queue retention",
        "subjects": ["jobs.>", "jobs.spawn.>", "jobs.priority.>"],
        "retention": RetentionPolicy.WORK_QUEUE,
        "max_msgs": 50_000_000,  # 50M messages
        "max_bytes": 500 * 1024 * 1024 * 1024,  # 500GB
        "max_msg_size": 64 * 1024 * 1024,  # 64MB
        "num_replicas": 3,
        "duplicate_window": 300_000_000_000,  # 5 minutes
    },
    "agentforge_results_million": {
        "description": "Million-scale results collection with limits retention",
        "subjects": ["results.>", "results.success.>", "results.error.>"],
        "retention": RetentionPolicy.LIMITS,
        "max_msgs": 100_000_000,  # 100M messages
        "max_bytes": 1024 * 1024 * 1024 * 1024,  # 1TB
        "max_msg_size": 32 * 1024 * 1024,  # 32MB
        "num_replicas": 3,
        "duplicate_window": 300_000_000_000,
    },
    "agentforge_neural_mesh": {
        "description": "Neural mesh memory synchronization at million-scale",
        "subjects": ["mesh.sync.>", "mesh.broadcast.>", "mesh.crdt.>"],
        "retention": RetentionPolicy.LIMITS,
        "max_msgs": 200_000_000,  # 200M messages
        "max_bytes": 2048 * 1024 * 1024 * 1024,  # 2TB
        "max_msg_size": 16 * 1024 * 1024,  # 16MB
        "num_replicas": 5,  # Higher replication for critical mesh data
        "duplicate_window": 600_000_000_000,  # 10 minutes
    },
    "agentforge_agent_lifecycle": {
        "description": "Million-agent lifecycle management",
        "subjects": ["agent.spawn.>", "agent.terminate.>", "agent.health.>", "agent.metrics.>"],
        "retention": RetentionPolicy.LIMITS,
        "max_msgs": 1_000_000_000,  # 1B messages for lifecycle events
        "max_bytes": 512 * 1024 * 1024 * 1024,  # 512GB
        "max_msg_size": 4 * 1024 * 1024,  # 4MB
        "num_replicas": 3,
        "duplicate_window": 120_000_000_000,  # 2 minutes
    },
    "agentforge_telemetry": {
        "description": "High-frequency telemetry and metrics",
        "subjects": ["telemetry.>", "metrics.>", "traces.>"],
        "retention": RetentionPolicy.LIMITS,
        "max_msgs": 500_000_000,  # 500M messages
        "max_bytes": 256 * 1024 * 1024 * 1024,  # 256GB
        "max_msg_size": 1024 * 1024,  # 1MB
        "num_replicas": 2,  # Lower replication for telemetry
        "duplicate_window": 60_000_000_000,  # 1 minute
    },
    "agentforge_orchestration": {
        "description": "High-level orchestration and coordination",
        "subjects": ["orchestration.>", "coordination.>", "consensus.>"],
        "retention": RetentionPolicy.LIMITS,
        "max_msgs": 10_000_000,  # 10M messages
        "max_bytes": 100 * 1024 * 1024 * 1024,  # 100GB
        "max_msg_size": 64 * 1024 * 1024,  # 64MB
        "num_replicas": 5,  # Highest replication for orchestration
        "duplicate_window": 1800_000_000_000,  # 30 minutes
    }
}

# Consumer templates for different workload patterns
CONSUMER_TEMPLATES = {
    "high_throughput": {
        "description": "High-throughput consumer for job processing",
        "max_ack_pending": 50000,
        "ack_wait": 300_000_000_000,  # 5 minutes
        "max_deliver": 3,
        "backoff": [1_000_000_000, 10_000_000_000, 60_000_000_000],  # 1s, 10s, 60s
    },
    "low_latency": {
        "description": "Low-latency consumer for real-time processing",
        "max_ack_pending": 1000,
        "ack_wait": 30_000_000_000,  # 30 seconds
        "max_deliver": 5,
        "backoff": [100_000_000, 1_000_000_000, 5_000_000_000],  # 100ms, 1s, 5s
    },
    "batch_processing": {
        "description": "Batch processing consumer with high pending limits",
        "max_ack_pending": 100000,
        "ack_wait": 1800_000_000_000,  # 30 minutes
        "max_deliver": 2,
        "backoff": [60_000_000_000, 300_000_000_000],  # 1min, 5min
    },
    "neural_mesh": {
        "description": "Neural mesh synchronization consumer",
        "max_ack_pending": 25000,
        "ack_wait": 600_000_000_000,  # 10 minutes
        "max_deliver": 5,
        "backoff": [1_000_000_000, 5_000_000_000, 30_000_000_000],  # 1s, 5s, 30s
    }
}

class MillionScaleJetStreamSetup:
    """Setup manager for million-scale JetStream infrastructure"""
    
    def __init__(self):
        self.nc: Optional[NATS] = None
        self.js = None
        self.setup_results = {
            "streams_created": [],
            "streams_updated": [],
            "consumers_created": [],
            "errors": []
        }
    
    async def connect(self) -> bool:
        """Connect to NATS cluster"""
        try:
            self.nc = nats.NATS()
            await self.nc.connect(
                servers=NATS_URLS,
                connect_timeout=30,
                allow_reconnect=True,
                max_reconnect_attempts=10,
                reconnect_time_wait=5,
                ping_interval=20,
                max_outstanding_pings=3,
                name=f"jetstream-setup-{ENVIRONMENT}"
            )
            self.js = self.nc.jetstream()
            
            # Verify cluster connectivity
            cluster_info = await self.nc.request("$SYS.REQ.SERVER.PING", timeout=10)
            log.info(f"Connected to NATS cluster: {cluster_info.data.decode()}")
            return True
            
        except Exception as e:
            log.error(f"Failed to connect to NATS cluster: {e}")
            self.setup_results["errors"].append(f"Connection failed: {e}")
            return False
    
    async def setup_streams(self) -> bool:
        """Create all million-scale streams"""
        log.info("Setting up million-scale streams...")
        success = True
        
        for stream_name, config in MILLION_SCALE_STREAMS.items():
            try:
                await self._create_or_update_stream(stream_name, config)
            except Exception as e:
                log.error(f"Failed to setup stream {stream_name}: {e}")
                self.setup_results["errors"].append(f"Stream {stream_name}: {e}")
                success = False
        
        return success
    
    async def _create_or_update_stream(self, stream_name: str, config: Dict):
        """Create or update a stream with million-scale configuration"""
        try:
            # Check if stream exists
            existing_stream = None
            try:
                existing_stream = await self.js.stream_info(stream_name)
                log.info(f"Stream {stream_name} already exists, checking configuration...")
            except Exception:
                pass
            
            # Create stream configuration
            stream_config = StreamConfig(
                name=stream_name,
                description=config["description"],
                subjects=config["subjects"],
                retention=config["retention"],
                storage=StorageType.FILE,
                max_consumers=-1,  # Unlimited consumers
                max_msgs=config["max_msgs"],
                max_bytes=config["max_bytes"],
                max_msg_size=config["max_msg_size"],
                num_replicas=config["num_replicas"],
                duplicate_window=config["duplicate_window"],
                discard=DiscardPolicy.OLD,
                allow_direct=True,
                allow_rollup_hdrs=True,
                deny_delete=True,  # Prevent accidental deletion
                deny_purge=False,  # Allow purging for maintenance
            )
            
            if existing_stream:
                # Update existing stream if needed
                if self._needs_update(existing_stream.config, stream_config):
                    await self.js.update_stream(stream_config)
                    log.info(f"Updated stream: {stream_name}")
                    self.setup_results["streams_updated"].append(stream_name)
                else:
                    log.info(f"Stream {stream_name} configuration is up to date")
            else:
                # Create new stream
                await self.js.add_stream(stream_config)
                log.info(f"Created stream: {stream_name}")
                self.setup_results["streams_created"].append(stream_name)
            
            # Verify stream health
            await self._verify_stream_health(stream_name)
            
        except Exception as e:
            log.error(f"Error creating/updating stream {stream_name}: {e}")
            raise
    
    def _needs_update(self, existing_config, new_config) -> bool:
        """Check if stream configuration needs updating"""
        # Compare key configuration parameters
        return (
            existing_config.max_msgs != new_config.max_msgs or
            existing_config.max_bytes != new_config.max_bytes or
            existing_config.num_replicas != new_config.num_replicas or
            set(existing_config.subjects) != set(new_config.subjects)
        )
    
    async def _verify_stream_health(self, stream_name: str):
        """Verify stream is healthy and replicated"""
        info = await self.js.stream_info(stream_name)
        
        # Check replica health
        if hasattr(info.cluster, 'replicas'):
            healthy_replicas = sum(1 for replica in info.cluster.replicas if replica.current)
            expected_replicas = info.config.num_replicas
            
            if healthy_replicas < expected_replicas:
                log.warning(f"Stream {stream_name} has {healthy_replicas}/{expected_replicas} healthy replicas")
            else:
                log.info(f"Stream {stream_name} has all {expected_replicas} replicas healthy")
    
    async def setup_default_consumers(self) -> bool:
        """Create default consumers for each stream"""
        log.info("Setting up default consumers...")
        success = True
        
        consumer_mappings = {
            "agentforge_jobs_million": [
                ("job_processor_high_throughput", "jobs.>", "high_throughput"),
                ("job_processor_priority", "jobs.priority.>", "low_latency"),
            ],
            "agentforge_results_million": [
                ("results_collector", "results.>", "batch_processing"),
                ("error_handler", "results.error.>", "low_latency"),
            ],
            "agentforge_neural_mesh": [
                ("mesh_synchronizer", "mesh.sync.>", "neural_mesh"),
                ("mesh_broadcaster", "mesh.broadcast.>", "low_latency"),
            ],
            "agentforge_agent_lifecycle": [
                ("lifecycle_manager", "agent.>", "high_throughput"),
                ("health_monitor", "agent.health.>", "low_latency"),
            ],
            "agentforge_telemetry": [
                ("telemetry_collector", "telemetry.>", "batch_processing"),
                ("metrics_processor", "metrics.>", "high_throughput"),
            ],
            "agentforge_orchestration": [
                ("orchestrator", "orchestration.>", "low_latency"),
                ("consensus_participant", "consensus.>", "low_latency"),
            ]
        }
        
        for stream_name, consumers in consumer_mappings.items():
            for consumer_name, filter_subject, template_name in consumers:
                try:
                    await self._create_consumer(stream_name, consumer_name, filter_subject, template_name)
                except Exception as e:
                    log.error(f"Failed to create consumer {consumer_name}: {e}")
                    self.setup_results["errors"].append(f"Consumer {consumer_name}: {e}")
                    success = False
        
        return success
    
    async def _create_consumer(self, stream_name: str, consumer_name: str, 
                             filter_subject: str, template_name: str):
        """Create a consumer with specified template"""
        template = CONSUMER_TEMPLATES[template_name]
        
        # Check if consumer already exists
        try:
            existing_consumer = await self.js.consumer_info(stream_name, consumer_name)
            log.info(f"Consumer {consumer_name} already exists")
            return
        except Exception:
            pass
        
        consumer_config = ConsumerConfig(
            durable_name=consumer_name,
            description=f"{template['description']} for {stream_name}",
            deliver_policy=DeliverPolicy.ALL,
            ack_policy=AckPolicy.EXPLICIT,
            replay_policy=ReplayPolicy.INSTANT,
            filter_subject=filter_subject,
            max_ack_pending=template["max_ack_pending"],
            ack_wait=template["ack_wait"],
            max_deliver=template["max_deliver"],
            backoff=template["backoff"],
            flow_control=True,
            idle_heartbeat=30_000_000_000,  # 30 seconds
        )
        
        await self.js.add_consumer(stream_name, consumer_config)
        log.info(f"Created consumer: {consumer_name} on stream {stream_name}")
        self.setup_results["consumers_created"].append(f"{stream_name}/{consumer_name}")
    
    async def create_monitoring_consumers(self) -> bool:
        """Create consumers for monitoring and observability"""
        log.info("Setting up monitoring consumers...")
        
        monitoring_consumers = [
            ("agentforge_telemetry", "prometheus_exporter", "telemetry.prometheus.>", "batch_processing"),
            ("agentforge_telemetry", "grafana_collector", "telemetry.grafana.>", "high_throughput"),
            ("agentforge_results_million", "slo_monitor", "results.>", "low_latency"),
            ("agentforge_agent_lifecycle", "capacity_planner", "agent.metrics.>", "batch_processing"),
        ]
        
        success = True
        for stream_name, consumer_name, filter_subject, template_name in monitoring_consumers:
            try:
                await self._create_consumer(stream_name, consumer_name, filter_subject, template_name)
            except Exception as e:
                log.error(f"Failed to create monitoring consumer {consumer_name}: {e}")
                success = False
        
        return success
    
    async def verify_setup(self) -> Dict:
        """Verify the complete setup and return status"""
        log.info("Verifying million-scale JetStream setup...")
        
        verification_results = {
            "cluster_info": {},
            "streams": {},
            "consumers": {},
            "performance_test": {},
            "recommendations": []
        }
        
        try:
            # Get cluster information
            server_info = await self.nc.request("$SYS.REQ.SERVER.INFO", timeout=10)
            verification_results["cluster_info"] = json.loads(server_info.data.decode())
            
            # Verify each stream
            for stream_name in MILLION_SCALE_STREAMS.keys():
                try:
                    stream_info = await self.js.stream_info(stream_name)
                    verification_results["streams"][stream_name] = {
                        "messages": stream_info.state.messages,
                        "bytes": stream_info.state.bytes,
                        "consumers": stream_info.state.consumer_count,
                        "replicas": getattr(stream_info.cluster, 'replicas', []) if hasattr(stream_info, 'cluster') else []
                    }
                except Exception as e:
                    verification_results["streams"][stream_name] = {"error": str(e)}
            
            # Run performance test
            verification_results["performance_test"] = await self._run_performance_test()
            
            # Generate recommendations
            verification_results["recommendations"] = self._generate_recommendations(verification_results)
            
        except Exception as e:
            log.error(f"Verification failed: {e}")
            verification_results["error"] = str(e)
        
        return verification_results
    
    async def _run_performance_test(self) -> Dict:
        """Run basic performance test"""
        log.info("Running performance test...")
        
        test_subject = "test.performance"
        test_messages = 1000
        test_payload = b"x" * 1024  # 1KB message
        
        try:
            # Publish test messages
            start_time = time.time()
            for i in range(test_messages):
                await self.js.publish(test_subject, test_payload)
            publish_time = time.time() - start_time
            
            publish_rate = test_messages / publish_time
            
            return {
                "messages_published": test_messages,
                "publish_time_seconds": publish_time,
                "publish_rate_msg_per_sec": publish_rate,
                "estimated_million_scale_time": (1_000_000 / publish_rate) if publish_rate > 0 else float('inf')
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_recommendations(self, verification_results: Dict) -> List[str]:
        """Generate setup recommendations based on verification results"""
        recommendations = []
        
        # Check cluster size
        cluster_info = verification_results.get("cluster_info", {})
        if "cluster" in cluster_info:
            cluster_size = len(cluster_info["cluster"].get("urls", []))
            if cluster_size < 3:
                recommendations.append(f"Cluster has only {cluster_size} nodes. Recommend minimum 3 nodes for HA.")
        
        # Check stream health
        for stream_name, stream_info in verification_results.get("streams", {}).items():
            if "error" in stream_info:
                recommendations.append(f"Stream {stream_name} has issues: {stream_info['error']}")
            elif stream_info.get("consumers", 0) == 0:
                recommendations.append(f"Stream {stream_name} has no consumers configured.")
        
        # Check performance
        perf_test = verification_results.get("performance_test", {})
        if "publish_rate_msg_per_sec" in perf_test:
            rate = perf_test["publish_rate_msg_per_sec"]
            if rate < 1000:
                recommendations.append(f"Low publish rate ({rate:.0f} msg/sec). Consider optimizing NATS configuration.")
            elif rate > 100000:
                recommendations.append(f"Excellent publish rate ({rate:.0f} msg/sec). System ready for million-scale.")
        
        if not recommendations:
            recommendations.append("Setup looks good! System appears ready for million-scale operations.")
        
        return recommendations
    
    async def cleanup_test_data(self):
        """Clean up test data created during setup"""
        try:
            # Remove test subjects/messages if any were created
            test_subjects = ["test.performance"]
            for subject in test_subjects:
                try:
                    await self.js.purge_stream("agentforge_telemetry", subject=subject)
                except Exception:
                    pass  # Ignore errors during cleanup
        except Exception as e:
            log.warning(f"Cleanup warning: {e}")
    
    async def close(self):
        """Close NATS connection"""
        if self.nc and not self.nc.is_closed:
            await self.nc.drain()
    
    def print_setup_summary(self):
        """Print setup summary"""
        print("\n" + "="*80)
        print("MILLION-SCALE JETSTREAM SETUP SUMMARY")
        print("="*80)
        
        print(f"Streams Created: {len(self.setup_results['streams_created'])}")
        for stream in self.setup_results['streams_created']:
            print(f"  ✓ {stream}")
        
        print(f"\nStreams Updated: {len(self.setup_results['streams_updated'])}")
        for stream in self.setup_results['streams_updated']:
            print(f"  ↻ {stream}")
        
        print(f"\nConsumers Created: {len(self.setup_results['consumers_created'])}")
        for consumer in self.setup_results['consumers_created']:
            print(f"  ✓ {consumer}")
        
        if self.setup_results['errors']:
            print(f"\nErrors: {len(self.setup_results['errors'])}")
            for error in self.setup_results['errors']:
                print(f"  ✗ {error}")
        
        print("\n" + "="*80)

async def main():
    """Main setup function"""
    print("AgentForge Million-Scale JetStream Setup")
    print("="*50)
    
    setup = MillionScaleJetStreamSetup()
    
    try:
        # Connect to cluster
        if not await setup.connect():
            print("❌ Failed to connect to NATS cluster")
            return 1
        
        print("✅ Connected to NATS cluster")
        
        # Setup streams
        if not await setup.setup_streams():
            print("❌ Failed to setup all streams")
        else:
            print("✅ All streams configured successfully")
        
        # Setup consumers
        if not await setup.setup_default_consumers():
            print("❌ Failed to setup all consumers")
        else:
            print("✅ All default consumers created")
        
        # Setup monitoring
        if not await setup.create_monitoring_consumers():
            print("❌ Failed to setup monitoring consumers")
        else:
            print("✅ Monitoring consumers created")
        
        # Verify setup
        verification = await setup.verify_setup()
        
        # Print results
        setup.print_setup_summary()
        
        print("\nVERIFICATION RESULTS:")
        print("-" * 30)
        
        if "performance_test" in verification:
            perf = verification["performance_test"]
            if "publish_rate_msg_per_sec" in perf:
                print(f"Publish Rate: {perf['publish_rate_msg_per_sec']:.0f} messages/second")
                print(f"Estimated time for 1M messages: {perf['estimated_million_scale_time']:.1f} seconds")
        
        print("\nRECOMMENDATIONS:")
        print("-" * 20)
        for rec in verification.get("recommendations", []):
            print(f"• {rec}")
        
        # Cleanup
        await setup.cleanup_test_data()
        
        print("\n🎉 Million-scale JetStream setup completed!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Setup interrupted by user")
        return 1
    except Exception as e:
        log.error(f"Setup failed with error: {e}")
        print(f"❌ Setup failed: {e}")
        return 1
    finally:
        await setup.close()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

