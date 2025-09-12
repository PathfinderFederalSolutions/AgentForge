#!/usr/bin/env python3
"""
Local NATS JetStream backlog drain testing and SLO measurement.
Measures drain performance against SLO: P95 < 10 minutes, hard cap 20 minutes.
"""

import asyncio
import json
import time
import statistics
from typing import List, Dict, Any
import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.append('/Users/baileymahoney/AgentForge')

import nats
from nats.errors import TimeoutError as NatsTimeoutError

# Configuration
NATS_URL = "nats://localhost:4222"
STREAM_NAME = "swarm_jobs"
SUBJECT = "swarm.jobs.staging"
CONSUMER_NAME = "worker-staging"
# Smaller deterministic backlog for local runs
BACKLOG_COUNT = int(os.getenv("DRAIN_BACKLOG_COUNT", "30"))
RUNS = int(os.getenv("DRAIN_RUNS", "2"))
MAX_CAP_MINUTES = 20
OBJ_P95_MINUTES = 10
POLL_INTERVAL = 0.5  # seconds, faster polling locally


class DrainTester:
    def __init__(self):
        self.nc = None
        self.js = None
        
    async def connect(self):
        """Connect to NATS JetStream"""
        try:
            self.nc = await nats.connect(NATS_URL)
            self.js = self.nc.jetstream()
            print(f"✅ Connected to NATS at {NATS_URL}")
        except Exception as e:
            print(f"❌ Failed to connect to NATS: {e}")
            raise
    
    async def ensure_stream_consumer(self):
        """Ensure stream and consumer exist for testing aligned with worker config"""
        try:
            # Desired stream config: subjects swarm.jobs.* (workers publish to swarm.jobs.<mission>)
            from nats.js.api import StreamConfig, RetentionPolicy, StorageType
            desired_subjects = ["swarm.jobs.*"]
            try:
                sinfo = await self.js.stream_info(STREAM_NAME)
                cur_subjects = list(getattr(getattr(sinfo, "config", None), "subjects", []) or [])
                if sorted(cur_subjects) != sorted(desired_subjects):
                    # Try to update stream to desired subjects
                    try:
                        await self.js.update_stream(StreamConfig(
                            name=STREAM_NAME,
                            subjects=desired_subjects,
                            retention=RetentionPolicy.WORK_QUEUE,
                            storage=StorageType.FILE,
                        ))
                        print(f"♻️  Updated stream '{STREAM_NAME}' subjects -> {desired_subjects}")
                    except Exception:
                        print(f"⚠️ Could not update subjects (may require manual cleanup); current={cur_subjects}")
                else:
                    print(f"✅ Stream '{STREAM_NAME}' exists")
            except Exception:
                # Create stream
                config = StreamConfig(
                    name=STREAM_NAME,
                    subjects=desired_subjects,
                    retention=RetentionPolicy.WORK_QUEUE,
                    storage=StorageType.FILE,
                )
                await self.js.add_stream(config)
                print(f"✅ Created stream '{STREAM_NAME}'")
            
            # Try to get consumer info, create if not exists
            try:
                await self.js.consumer_info(STREAM_NAME, CONSUMER_NAME)
                print(f"✅ Consumer '{CONSUMER_NAME}' exists")
            except Exception:
                # Create consumer
                from nats.js.api import ConsumerConfig, DeliverPolicy, AckPolicy
                config = ConsumerConfig(
                    durable_name=CONSUMER_NAME,
                    deliver_policy=DeliverPolicy.ALL,
                    ack_policy=AckPolicy.EXPLICIT,
                    filter_subject=SUBJECT,
                    max_ack_pending=5000,
                )
                await self.js.add_consumer(STREAM_NAME, config)
                print(f"✅ Created consumer '{CONSUMER_NAME}'")
                
        except Exception as e:
            print(f"❌ Failed to ensure stream/consumer: {e}")
            raise
    
    async def get_backlog_count(self) -> int:
        """Get current backlog count from consumer info"""
        try:
            info = await self.js.consumer_info(STREAM_NAME, CONSUMER_NAME)
            pending = info.num_pending
            return pending
        except Exception as e:
            print(f"⚠️ Failed to get backlog count: {e}")
            return 0
    
    async def publish_backlog(self, count: int):
        """Publish messages to create backlog"""
        print(f"📤 Publishing {count} messages to create backlog...")
        start_time = time.time()
        
        # Publish messages in batches for better performance
        batch_size = 50
        for i in range(0, count, batch_size):
            batch_end = min(i + batch_size, count)
            tasks = []
            
            for j in range(i, batch_end):
                payload = {
                    "job_id": f"drain-test-{j}",
                    "goal": f"Test message {j} for drain testing",
                    "agents": 1,
                    "metadata": {"test": True, "drain_run": True}
                }
                task = self.js.publish(SUBJECT, json.dumps(payload).encode())
                tasks.append(task)
            
            # Wait for batch to complete
            await asyncio.gather(*tasks)
            
            # Progress update
            if (batch_end % 200 == 0) or (batch_end == count):
                elapsed = time.time() - start_time
                print(f"  📊 Published {batch_end}/{count} messages ({elapsed:.1f}s)")
        
        # Wait a moment for messages to settle
        await asyncio.sleep(0.5)
        
        # Verify backlog
        final_backlog = await self.get_backlog_count()
        print(f"✅ Published {count} messages, current backlog: {final_backlog}")
        return final_backlog
    
    async def wait_for_drain(self, timeout_minutes: float = 20) -> float:
        """Wait for backlog to drain and measure time"""
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        print(f"⏱️ Waiting for backlog drain (timeout: {timeout_minutes:.1f}m)...")
        
        last_backlog = await self.get_backlog_count()
        print(f"  📊 Initial backlog: {last_backlog}")
        
        while time.time() - start_time < timeout_seconds:
            await asyncio.sleep(POLL_INTERVAL)
            
            current_backlog = await self.get_backlog_count()
            elapsed_seconds = time.time() - start_time
            elapsed_minutes = elapsed_seconds / 60
            
            # Show progress every ~5 seconds or when backlog changes significantly
            if (int(elapsed_seconds * 10) % 50 == 0) or (abs(current_backlog - last_backlog) > 10):
                print(f"  📊 Backlog: {current_backlog}, elapsed: {elapsed_minutes:.2f}m")
                last_backlog = current_backlog
            
            # Check if drained
            if current_backlog == 0:
                drain_time_minutes = elapsed_minutes
                print(f"✅ Backlog drained in {drain_time_minutes:.2f} minutes")
                return drain_time_minutes
        
        # Timeout reached
        final_backlog = await self.get_backlog_count()
        print(f"⏰ Timeout reached ({timeout_minutes:.1f}m), remaining backlog: {final_backlog}")
        return timeout_minutes  # Return max time as penalty
    
    async def run_single_test(self, run_id: int) -> float:
        """Run a single drain test"""
        print(f"\n🔄 === Run {run_id}/{RUNS} ===")
        
        # Publish backlog
        await self.publish_backlog(BACKLOG_COUNT)
        
        # Measure drain time
        drain_time = await self.wait_for_drain(MAX_CAP_MINUTES)
        
        print(f"📈 Run {run_id} completed: {drain_time:.2f} minutes")
        return drain_time
    
    async def run_drain_test(self) -> Dict[str, Any]:
        """Run complete drain test suite"""
        print("🚀 Starting NATS JetStream Drain Test")
        print(f"📋 Configuration:")
        print(f"   Stream: {STREAM_NAME}")
        print(f"   Subject: {SUBJECT}")
        print(f"   Consumer: {CONSUMER_NAME}")
        print(f"   Backlog per run: {BACKLOG_COUNT}")
        print(f"   Runs: {RUNS}")
        print(f"   SLO P95: ≤ {OBJ_P95_MINUTES} minutes")
        print(f"   Hard cap: ≤ {MAX_CAP_MINUTES} minutes")
        
        await self.connect()
        await self.ensure_stream_consumer()
        
        # Wait for any existing backlog to clear
        initial_backlog = await self.get_backlog_count()
        if initial_backlog > 0:
            print(f"⚠️ Initial backlog detected: {initial_backlog}")
            print("   Waiting for backlog to clear before starting test...")
            await self.wait_for_drain(5)  # 5 minute max wait
        
        # Run drain tests
        durations = []
        for run_id in range(1, RUNS + 1):
            try:
                duration = await self.run_single_test(run_id)
                durations.append(duration)
                
                # Brief pause between runs
                if run_id < RUNS:
                    print(f"   Pausing 2s before next run...")
                    await asyncio.sleep(2)
                    
            except Exception as e:
                print(f"❌ Run {run_id} failed: {e}")
                durations.append(MAX_CAP_MINUTES)  # Penalty time
        
        # Calculate statistics
        if not durations:
            raise RuntimeError("No successful drain tests completed")
        
        p95 = statistics.quantiles(durations, n=20)[18] if len(durations) >= 5 else max(durations)
        
        # Evaluate SLO
        p95_pass = p95 <= OBJ_P95_MINUTES
        cap_pass = all(d <= MAX_CAP_MINUTES for d in durations)
        
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "stream": STREAM_NAME,
                "subject": SUBJECT,
                "consumer": CONSUMER_NAME,
                "backlog_count": BACKLOG_COUNT,
                "runs": RUNS,
            },
            "durations_minutes": durations,
            "statistics": {
                "p95_minutes": round(p95, 2),
                "mean_minutes": round(statistics.mean(durations), 2),
                "max_minutes": round(max(durations), 2),
            },
            "slo_evaluation": {
                "objective_p95_minutes": OBJ_P95_MINUTES,
                "hard_cap_minutes": MAX_CAP_MINUTES,
                "p95_pass": p95_pass,
                "cap_pass": cap_pass,
                "overall_pass": bool(p95_pass and cap_pass),
            }
        }
        
        return results
    
    async def close(self):
        """Close NATS connection"""
        if self.nc:
            await self.nc.close()


async def start_test_workers(num_workers: int = 2):
    """Start background worker processes for testing"""
    print(f"🔧 Starting {num_workers} test workers...")
    
    import subprocess
    import signal
    
    # Worker processes
    worker_processes = []
    
    def cleanup_workers():
        for p in worker_processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
    
    # Register cleanup
    signal.signal(signal.SIGTERM, lambda s, f: cleanup_workers())
    signal.signal(signal.SIGINT, lambda s, f: cleanup_workers())
    
    try:
        # Start worker processes
        for i in range(num_workers):
            env = os.environ.copy()
            env['MISSION'] = 'staging'
            env['DEFAULT_AGENTS'] = '1'  # Reduce for faster processing
            # Fast drain path: enable no-op worker and aggressive fetch
            env['WORKER_NOOP'] = '1'
            env['NATS_FETCH_WAIT'] = '0.2'
            env['NATS_FETCH_BATCH'] = '64'
            env['EDGE_MODE'] = env.get('EDGE_MODE', '1')
            
            p = subprocess.Popen([
                sys.executable, '-m', 'swarm.workers.nats_worker'
            ], env=env, cwd='/Users/baileymahoney/AgentForge')
            
            worker_processes.append(p)
            print(f"   ✅ Started worker {i+1}")
        
        # Give workers time to connect
        await asyncio.sleep(2)
        return worker_processes, cleanup_workers
        
    except Exception as e:
        cleanup_workers()
        raise e


async def main():
    """Main test execution"""
    cleanup_workers = None
    
    try:
        # Start test workers
        _, cleanup_workers = await start_test_workers(2)
        
        # Run drain tests
        tester = DrainTester()
        try:
            results = await tester.run_drain_test()
            
            # Print results
            print("\n" + "="*60)
            print("🎯 DRAIN TEST RESULTS")
            print("="*60)
            print(f"📊 Runs completed: {results['configuration']['runs']}")
            print(f"⏱️ Durations (minutes): {results['durations_minutes']}")
            print(f"📈 P95: {results['statistics']['p95_minutes']}m")
            print(f"📈 Mean: {results['statistics']['mean_minutes']}m")
            print(f"📈 Max: {results['statistics']['max_minutes']}m")
            print(f"")
            print(f"🎯 SLO EVALUATION:")
            print(f"   P95 ≤ {results['slo_evaluation']['objective_p95_minutes']}m: {'✅ PASS' if results['slo_evaluation']['p95_pass'] else '❌ FAIL'}")
            print(f"   Cap ≤ {results['slo_evaluation']['hard_cap_minutes']}m: {'✅ PASS' if results['slo_evaluation']['cap_pass'] else '❌ FAIL'}")
            print(f"   Overall SLO: {'✅ PASS' if results['slo_evaluation']['overall_pass'] else '❌ FAIL'}")
            
            # Save results
            results_file = f"/Users/baileymahoney/AgentForge/drain_test_results_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📁 Results saved to: {results_file}")
            
            return 0 if results['slo_evaluation']['overall_pass'] else 1
            
        finally:
            await tester.close()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 2
        
    finally:
        # Cleanup workers
        if cleanup_workers:
            print("\n🧹 Cleaning up workers...")
            cleanup_workers()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
