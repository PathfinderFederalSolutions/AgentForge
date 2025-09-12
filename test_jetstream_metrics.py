#!/usr/bin/env python3
"""
Test script to create JetStream streams and publish messages to verify quantity tracking
"""
import asyncio
import json
import time
import os
import sys

try:
    import nats
    from nats.js.api import StreamConfig
    NATS_AVAILABLE = True
except ImportError:
    print("NATS Python client not available")
    NATS_AVAILABLE = False

async def test_jetstream_operations():
    """Test JetStream operations and verify metrics"""
    if not NATS_AVAILABLE:
        print("❌ NATS Python client not available")
        return False
    
    try:
        # Connect to NATS
        nc = await nats.connect("nats://localhost:4222")
        js = nc.jetstream()
        
        print("✅ Connected to NATS server")
        
        # Create a test stream
        stream_config = StreamConfig(
            name="test_metrics_stream",
            subjects=["test.metrics.>"],
            retention="workqueue",
            storage="file",
            max_msgs=1000
        )
        
        try:
            await js.add_stream(stream_config)
            print("✅ Created test stream: test_metrics_stream")
        except Exception as e:
            if "stream name already in use" in str(e).lower():
                print("ℹ️  Stream already exists: test_metrics_stream")
            else:
                print(f"❌ Error creating stream: {e}")
                return False
        
        # Publish some test messages
        for i in range(10):
            subject = f"test.metrics.msg.{i}"
            payload = json.dumps({
                "id": i,
                "timestamp": time.time(),
                "test_data": f"Message {i} for metrics testing"
            })
            
            await js.publish(subject, payload.encode())
            print(f"✅ Published message {i+1}/10 to {subject}")
            await asyncio.sleep(0.1)  # Small delay between messages
        
        print("✅ Published 10 test messages to JetStream")
        
        # Wait a moment for metrics to update
        await asyncio.sleep(2)
        
        await nc.close()
        return True
        
    except Exception as e:
        print(f"❌ Error in JetStream operations: {e}")
        return False

def check_metrics_after_activity():
    """Check metrics after creating activity"""
    import requests
    
    try:
        # Check NATS exporter metrics
        response = requests.get("http://localhost:7777/metrics", timeout=5)
        metrics_text = response.text
        
        print("\n=== JetStream Metrics After Activity ===")
        
        # Extract key metrics
        lines = metrics_text.split('\n')
        jetstream_metrics = {}
        
        for line in lines:
            if line.startswith('jetstream_server_total_streams{') and not line.startswith('#'):
                value = line.split()[-1]
                jetstream_metrics['total_streams'] = int(float(value))
                
            elif line.startswith('jetstream_server_total_messages{') and not line.startswith('#'):
                value = line.split()[-1]
                jetstream_metrics['total_messages'] = int(float(value))
                
            elif line.startswith('jetstream_server_total_message_bytes{') and not line.startswith('#'):
                value = line.split()[-1]
                jetstream_metrics['total_bytes'] = int(float(value))
                
            elif line.startswith('jetstream_server_total_consumers{') and not line.startswith('#'):
                value = line.split()[-1]
                jetstream_metrics['total_consumers'] = int(float(value))
        
        print(f"📊 Total Streams: {jetstream_metrics.get('total_streams', 0)}")
        print(f"📊 Total Messages: {jetstream_metrics.get('total_messages', 0)}")
        print(f"📊 Total Bytes: {jetstream_metrics.get('total_bytes', 0)}")
        print(f"📊 Total Consumers: {jetstream_metrics.get('total_consumers', 0)}")
        
        # Verify quantity tracking
        if jetstream_metrics.get('total_streams', 0) > 0:
            print("✅ JetStream stream quantity tracking working")
        else:
            print("❌ No streams detected in metrics")
            
        if jetstream_metrics.get('total_messages', 0) > 0:
            print("✅ JetStream message quantity tracking working")
        else:
            print("⚠️  No messages detected in metrics (may need more time)")
        
        return jetstream_metrics
        
    except Exception as e:
        print(f"❌ Error checking metrics: {e}")
        return {}

async def main():
    print("🚀 Testing JetStream Quantity Tracking...")
    print("=" * 50)
    
    # Test JetStream operations
    success = await test_jetstream_operations()
    
    if success:
        print("\n⏳ Waiting for metrics to update...")
        await asyncio.sleep(3)
        
        # Check metrics
        metrics = check_metrics_after_activity()
        
        print("\n" + "=" * 50)
        if metrics.get('total_streams', 0) > 0 or metrics.get('total_messages', 0) > 0:
            print("✅ JetStream quantity tracking verification PASSED")
        else:
            print("❌ JetStream quantity tracking verification FAILED")
    else:
        print("❌ JetStream operations failed")

if __name__ == "__main__":
    asyncio.run(main())
