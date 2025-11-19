#!/usr/bin/env python3
"""
Production JetStream Setup - Create streams and consumers for AgentForge
"""
import asyncio

try:
    import nats
    from nats.js.api import StreamConfig, ConsumerConfig
    NATS_AVAILABLE = True
except ImportError:
    print("❌ NATS Python client not available")
    NATS_AVAILABLE = False

async def create_production_streams():
    """Create production JetStream streams for AgentForge"""
    if not NATS_AVAILABLE:
        return False
    
    try:
        nc = await nats.connect("nats://localhost:4222")
        js = nc.jetstream()
        
        print("🚀 Setting up production JetStream streams...")
        
        # 1. Swarm Jobs Stream
        jobs_stream = StreamConfig(
            name="swarm_jobs",
            subjects=["swarm.jobs.*"],
            retention="workqueue",
            storage="file"
        )
        
        try:
            await js.add_stream(jobs_stream)
            print("✅ Created stream: swarm_jobs")
        except Exception as e:
            if "stream name already in use" in str(e).lower():
                print("ℹ️  Stream already exists: swarm_jobs")
            else:
                print(f"❌ Error creating swarm_jobs stream: {e}")
                return False
        
        # 2. Swarm Results Stream
        results_stream = StreamConfig(
            name="swarm_results",
            subjects=["swarm.results.*"],
            retention="limits",
            storage="file"
        )
        
        try:
            await js.add_stream(results_stream)
            print("✅ Created stream: swarm_results")
        except Exception as e:
            if "stream name already in use" in str(e).lower():
                print("ℹ️  Stream already exists: swarm_results")
            else:
                print(f"❌ Error creating swarm_results stream: {e}")
                return False
        
        # 3. HITL (Human-in-the-Loop) Stream
        hitl_stream = StreamConfig(
            name="swarm_hitl",
            subjects=["swarm.hitl.*"],
            retention="limits",
            storage="file"
        )
        
        try:
            await js.add_stream(hitl_stream)
            print("✅ Created stream: swarm_hitl")
        except Exception as e:
            if "stream name already in use" in str(e).lower():
                print("ℹ️  Stream already exists: swarm_hitl")
            else:
                print(f"❌ Error creating swarm_hitl stream: {e}")
                return False
        
        await nc.close()
        return True
        
    except Exception as e:
        print(f"❌ Error setting up streams: {e}")
        return False

async def create_production_consumers():
    """Create production consumers for each environment"""
    if not NATS_AVAILABLE:
        return False
    
    try:
        nc = await nats.connect("nats://localhost:4222")
        js = nc.jetstream()
        
        print("🔄 Setting up production consumers...")
        
        environments = ["staging", "production", "development"]
        
        for env in environments:
            # Job consumers
            job_consumer = ConsumerConfig(
                durable_name=f"worker-{env}",
                deliver_policy="all",
                ack_policy="explicit",
                filter_subject=f"swarm.jobs.{env}"
            )
            
            try:
                await js.add_consumer("swarm_jobs", job_consumer)
                print(f"✅ Created consumer: worker-{env}")
            except Exception as e:
                if "consumer name already in use" in str(e).lower():
                    print(f"ℹ️  Consumer already exists: worker-{env}")
                else:
                    print(f"❌ Error creating worker-{env} consumer: {e}")
            
            # Results consumers
            results_consumer = ConsumerConfig(
                durable_name=f"results-{env}",
                deliver_policy="all",
                ack_policy="explicit",
                filter_subject=f"swarm.results.{env}"
            )
            
            try:
                await js.add_consumer("swarm_results", results_consumer)
                print(f"✅ Created consumer: results-{env}")
            except Exception as e:
                if "consumer name already in use" in str(e).lower():
                    print(f"ℹ️  Consumer already exists: results-{env}")
                else:
                    print(f"❌ Error creating results-{env} consumer: {e}")
            
            # HITL consumers
            hitl_consumer = ConsumerConfig(
                durable_name=f"hitl-{env}",
                deliver_policy="all",
                ack_policy="explicit",
                filter_subject=f"swarm.hitl.{env}"
            )
            
            try:
                await js.add_consumer("swarm_hitl", hitl_consumer)
                print(f"✅ Created consumer: hitl-{env}")
            except Exception as e:
                if "consumer name already in use" in str(e).lower():
                    print(f"ℹ️  Consumer already exists: hitl-{env}")
                else:
                    print(f"❌ Error creating hitl-{env} consumer: {e}")
        
        await nc.close()
        return True
        
    except Exception as e:
        print(f"❌ Error setting up consumers: {e}")
        return False

async def verify_setup():
    """Verify the setup by checking stream and consumer counts"""
    try:
        nc = await nats.connect("nats://localhost:4222")
        js = nc.jetstream()
        
        print("\n📊 Verifying JetStream setup...")
        
        # Get stream info
        streams = await js.streams_info()
        print(f"Total streams: {len(streams)}")
        
        for stream in streams:
            consumers_info = await js.consumers_info(stream.config.name)
            print(f"  - {stream.config.name}: {len(consumers_info)} consumers, {stream.state.messages} messages")
        
        await nc.close()
        return True
        
    except Exception as e:
        print(f"❌ Error verifying setup: {e}")
        return False

async def main():
    print("🚀 AgentForge Production JetStream Setup")
    print("=" * 50)
    
    if not NATS_AVAILABLE:
        print("❌ NATS client not available. Install with: pip install nats-py")
        return False
    
    # Create streams
    streams_ok = await create_production_streams()
    if not streams_ok:
        print("❌ Failed to create streams")
        return False
    
    # Create consumers
    consumers_ok = await create_production_consumers()
    if not consumers_ok:
        print("❌ Failed to create consumers")
        return False
    
    # Verify setup
    verify_ok = await verify_setup()
    if not verify_ok:
        print("❌ Failed to verify setup")
        return False
    
    print("\n✅ Production JetStream setup complete!")
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    if not success:
        exit(1)
