#!/usr/bin/env python3
"""
Alert scenario testing for SLO monitoring.
Tests sustained backlog, exporter down, and ack pending scenarios.
"""

import asyncio
import json
import subprocess
import time
import sys
sys.path.append('/Users/baileymahoney/AgentForge')

import nats
from typing import List

class AlertScenarioTester:
    def __init__(self):
        self.nc = None
        self.js = None
        
    async def connect(self):
        """Connect to NATS JetStream"""
        self.nc = await nats.connect("nats://localhost:4222")
        self.js = self.nc.jetstream()
        print("✅ Connected to NATS JetStream")
    
    async def test_sustained_backlog(self, message_count: int = 3500):
        """Test A: Create sustained backlog of ≥3000 messages for >10m"""
        print(f"\n🧪 TEST A: Sustained Backlog ({message_count} messages)")
        print("   Expected: Warning after 10m, Critical after 15m")
        
        # Scale workers to 0 to prevent processing
        print("   1. Scaling workers to 0 to prevent message processing...")
        try:
            subprocess.run([
                "kubectl", "-n", "agentforge-staging", 
                "scale", "deploy/nats-worker", "--replicas=0"
            ], check=True, capture_output=True)
            print("      ✅ Workers scaled to 0")
        except subprocess.CalledProcessError as e:
            print(f"      ⚠️ Failed to scale workers: {e}")
        
        # Create sustained backlog
        print(f"   2. Publishing {message_count} messages...")
        start_time = time.time()
        
        for i in range(0, message_count, 100):
            batch_end = min(i + 100, message_count)
            tasks = []
            
            for j in range(i, batch_end):
                payload = {
                    "job_id": f"sustained-test-{j}",
                    "goal": f"Sustained backlog test message {j}",
                    "agents": 1,
                    "metadata": {"test_type": "sustained_backlog"}
                }
                task = self.js.publish("swarm.jobs.staging", json.dumps(payload).encode())
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            if (batch_end % 500 == 0) or (batch_end == message_count):
                elapsed = time.time() - start_time
                print(f"      Published {batch_end}/{message_count} messages ({elapsed:.1f}s)")
        
        # Check backlog
        try:
            info = await self.js.consumer_info("swarm_jobs", "worker-staging")
            current_backlog = info.num_pending
            print(f"   3. Current backlog: {current_backlog} messages")
            
            if current_backlog >= 3000:
                print("   ✅ Sustained backlog condition met (≥3000 messages)")
                print("   📊 Monitor Prometheus alerts:")
                print("      - Warning alert should fire after 10 minutes")
                print("      - Critical alert should fire after 15 minutes")
                print(f"      - Check: http://localhost:9090/alerts")
                
                return {
                    "test": "sustained_backlog", 
                    "status": "setup_complete",
                    "backlog": current_backlog,
                    "messages_published": message_count,
                    "monitor_duration": "15+ minutes"
                }
            else:
                print(f"   ❌ Backlog too low: {current_backlog} < 3000")
                return {"test": "sustained_backlog", "status": "failed", "backlog": current_backlog}
                
        except Exception as e:
            print(f"   ❌ Failed to check backlog: {e}")
            return {"test": "sustained_backlog", "status": "error", "error": str(e)}
    
    def test_exporter_down(self):
        """Test B: Scale NATS exporter to 0, then back to 1"""
        print(f"\n🧪 TEST B: Exporter Down Scenario")
        print("   Expected: Alert fires when exporter is down")
        
        try:
            # Scale exporter down
            print("   1. Scaling NATS Prometheus exporter to 0...")
            subprocess.run([
                "kubectl", "-n", "agentforge-staging", 
                "scale", "deploy/nats-prometheus-exporter", "--replicas=0"
            ], check=True, capture_output=True)
            print("      ✅ Exporter scaled to 0")
            
            # Wait a moment for alert to fire
            print("   2. Waiting 2 minutes for alert to fire...")
            time.sleep(120)
            
            # Scale back up
            print("   3. Scaling exporter back to 1...")
            subprocess.run([
                "kubectl", "-n", "agentforge-staging", 
                "scale", "deploy/nats-prometheus-exporter", "--replicas=1"
            ], check=True, capture_output=True)
            print("      ✅ Exporter scaled back to 1")
            
            return {
                "test": "exporter_down", 
                "status": "completed",
                "duration": "2 minutes",
                "alert_expected": "NATSServerDown"
            }
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to scale exporter: {e}")
            return {"test": "exporter_down", "status": "error", "error": str(e)}
    
    async def test_ack_pending_rise(self, message_count: int = 500):
        """Test C: Scale workers to 0 after they pull messages to increase ack pending"""
        print(f"\n🧪 TEST C: Ack Pending Rise Scenario")
        print("   Expected: Ack pending grows when workers can't acknowledge")
        
        try:
            # Ensure workers are running first
            print("   1. Ensuring workers are running...")
            subprocess.run([
                "kubectl", "-n", "agentforge-staging", 
                "scale", "deploy/nats-worker", "--replicas=2"
            ], check=True, capture_output=True)
            time.sleep(10)  # Let workers start
            
            # Publish messages
            print(f"   2. Publishing {message_count} messages...")
            start_time = time.time()
            
            for i in range(message_count):
                payload = {
                    "job_id": f"ack-pending-test-{i}",
                    "goal": f"Ack pending test message {i}",
                    "agents": 1,
                    "metadata": {"test_type": "ack_pending"}
                }
                await self.js.publish("swarm.jobs.staging", json.dumps(payload).encode())
                
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"      Published {i+1}/{message_count} messages ({elapsed:.1f}s)")
            
            # Let workers start processing
            print("   3. Letting workers start processing (30 seconds)...")
            time.sleep(30)
            
            # Scale workers to 0 while they have unacknowledged messages
            print("   4. Scaling workers to 0 while processing...")
            subprocess.run([
                "kubectl", "-n", "agentforge-staging", 
                "scale", "deploy/nats-worker", "--replicas=0"
            ], check=True, capture_output=True)
            
            # Check ack pending
            print("   5. Checking ack pending count...")
            try:
                info = await self.js.consumer_info("swarm_jobs", "worker-staging")
                ack_pending = info.ack_pending
                print(f"      Ack pending: {ack_pending}")
                
                if ack_pending > 0:
                    print("   ✅ Ack pending rise condition created")
                    print("   📊 Monitor for AckPendingHigh alert")
                    
                    # Scale workers back up to clear
                    print("   6. Scaling workers back up to clear pending acks...")
                    subprocess.run([
                        "kubectl", "-n", "agentforge-staging", 
                        "scale", "deploy/nats-worker", "--replicas=2"
                    ], check=True, capture_output=True)
                    
                    return {
                        "test": "ack_pending_rise",
                        "status": "completed", 
                        "ack_pending": ack_pending,
                        "messages_published": message_count
                    }
                else:
                    print("   ⚠️ No ack pending detected")
                    return {"test": "ack_pending_rise", "status": "no_pending", "ack_pending": 0}
                    
            except Exception as e:
                print(f"   ❌ Failed to check ack pending: {e}")
                return {"test": "ack_pending_rise", "status": "error", "error": str(e)}
                
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to manage workers: {e}")
            return {"test": "ack_pending_rise", "status": "error", "error": str(e)}
    
    async def cleanup(self):
        """Cleanup: Restore normal state"""
        print(f"\n🧹 CLEANUP: Restoring normal state")
        
        try:
            # Scale workers back to normal
            subprocess.run([
                "kubectl", "-n", "agentforge-staging", 
                "scale", "deploy/nats-worker", "--replicas=2"
            ], check=True, capture_output=True)
            print("   ✅ Workers scaled back to 2")
            
            # Ensure exporter is running
            subprocess.run([
                "kubectl", "-n", "agentforge-staging", 
                "scale", "deploy/nats-prometheus-exporter", "--replicas=1"
            ], check=True, capture_output=True)
            print("   ✅ Exporter ensured running")
            
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ Cleanup warning: {e}")
        
        if self.nc:
            await self.nc.close()
            print("   ✅ NATS connection closed")
    
    async def run_all_tests(self):
        """Run all alert scenario tests"""
        print("🚀 Starting Alert Scenario Testing")
        results = []
        
        await self.connect()
        
        try:
            # Test A: Sustained backlog
            result_a = await self.test_sustained_backlog()
            results.append(result_a)
            
            # Test B: Exporter down
            result_b = self.test_exporter_down()
            results.append(result_b)
            
            # Test C: Ack pending rise
            result_c = await self.test_ack_pending_rise()
            results.append(result_c)
            
        finally:
            await self.cleanup()
        
        return results


async def main():
    """Main test execution"""
    tester = AlertScenarioTester()
    
    try:
        results = await tester.run_all_tests()
        
        print("\n" + "="*60)
        print("🎯 ALERT SCENARIO TEST RESULTS")
        print("="*60)
        
        for result in results:
            test_name = result.get("test", "unknown")
            status = result.get("status", "unknown")
            print(f"📊 {test_name}: {status.upper()}")
            if "error" in result:
                print(f"   ❌ Error: {result['error']}")
            elif test_name == "sustained_backlog" and status == "setup_complete":
                print(f"   📈 Backlog: {result['backlog']} messages")
                print(f"   ⏱️ Monitor alerts for {result['monitor_duration']}")
            elif test_name == "exporter_down" and status == "completed":
                print(f"   ⏱️ Duration: {result['duration']}")
                print(f"   🚨 Expected alert: {result['alert_expected']}")
            elif test_name == "ack_pending_rise" and status == "completed":
                print(f"   📈 Ack pending: {result['ack_pending']}")
        
        # Save results
        results_file = f"/Users/baileymahoney/AgentForge/alert_scenario_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "tests": results,
                "summary": {
                    "total_tests": len(results),
                    "completed": len([r for r in results if r.get("status") in ["completed", "setup_complete"]]),
                    "errors": len([r for r in results if r.get("status") == "error"])
                }
            }, f, indent=2)
        
        print(f"📁 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
