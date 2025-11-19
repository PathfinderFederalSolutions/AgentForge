#!/usr/bin/env python3
"""
JetStream Backlog Stats and Health Check Script
"""
import asyncio
import requests
import json
from datetime import datetime

def check_prometheus_metrics():
    """Check JetStream metrics from Prometheus/NATS exporter"""
    print("📊 JetStream Metrics from NATS Exporter:")
    print("-" * 40)
    
    try:
        response = requests.get("http://localhost:7777/metrics", timeout=5)
        metrics_text = response.text
        
        # Parse key metrics
        metrics = {}
        for line in metrics_text.split('\n'):
            if 'jetstream_server_total_streams{' in line and not line.startswith('#'):
                metrics['total_streams'] = int(float(line.split()[-1]))
            elif 'jetstream_server_total_consumers{' in line and not line.startswith('#'):
                metrics['total_consumers'] = int(float(line.split()[-1]))
            elif 'jetstream_server_total_messages{' in line and not line.startswith('#'):
                metrics['total_messages'] = int(float(line.split()[-1]))
            elif 'jetstream_server_total_message_bytes{' in line and not line.startswith('#'):
                metrics['total_bytes'] = int(float(line.split()[-1]))
            elif 'jetstream_server_max_memory{' in line and not line.startswith('#'):
                metrics['max_memory'] = int(float(line.split()[-1]))
            elif 'jetstream_server_max_storage{' in line and not line.startswith('#'):
                metrics['max_storage'] = int(float(line.split()[-1]))
        
        print(f"✅ Total Streams: {metrics.get('total_streams', 0)}")
        print(f"✅ Total Consumers: {metrics.get('total_consumers', 0)}")
        print(f"📈 Total Messages: {metrics.get('total_messages', 0)}")
        print(f"💾 Total Bytes: {metrics.get('total_bytes', 0):,}")
        print(f"🧠 Max Memory: {metrics.get('max_memory', 0):,} bytes")
        print(f"💽 Max Storage: {metrics.get('max_storage', 0):,} bytes")
        
        # Calculate utilization
        if metrics.get('total_bytes', 0) > 0 and metrics.get('max_memory', 0) > 0:
            memory_util = (metrics['total_bytes'] / metrics['max_memory']) * 100
            print(f"📊 Memory Utilization: {memory_util:.4f}%")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error getting metrics: {e}")
        return {}

def check_nats_direct():
    """Check NATS JetStream status directly via HTTP API"""
    print("\n🔍 Direct NATS JetStream Status:")
    print("-" * 40)
    
    try:
        response = requests.get("http://localhost:8222/jsz", timeout=5)
        if response.status_code == 200:
            js_info = response.json()
            
            print(f"✅ JetStream Enabled: {not js_info.get('disabled', True)}")
            print(f"📊 Streams: {js_info.get('streams', 0)}")
            print(f"👥 Consumers: {js_info.get('consumers', 0)}")
            print(f"💾 Memory Used: {js_info.get('memory', 0):,} bytes")
            print(f"💽 Storage Used: {js_info.get('storage', 0):,} bytes")
            print(f"🏗️ Storage Directory: {js_info.get('config', {}).get('store_dir', 'N/A')}")
            
            return js_info
        else:
            print(f"❌ HTTP {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Error checking NATS: {e}")
        return {}

async def check_stream_details():
    """Get detailed stream information using NATS client"""
    print("\n📋 Detailed Stream Information:")
    print("-" * 40)
    
    try:
        import nats
        nc = await nats.connect("nats://localhost:4222")
        js = nc.jetstream()
        
        streams = await js.streams_info()
        
        for stream in streams:
            print(f"\n🌊 Stream: {stream.config.name}")
            print(f"   Subjects: {', '.join(stream.config.subjects)}")
            print(f"   Messages: {stream.state.messages}")
            print(f"   Bytes: {stream.state.bytes:,}")
            print(f"   First Seq: {stream.state.first_seq}")
            print(f"   Last Seq: {stream.state.last_seq}")
            print(f"   Retention: {stream.config.retention}")
            print(f"   Storage: {stream.config.storage}")
            
            # Get consumers for this stream
            consumers = await js.consumers_info(stream.config.name)
            print(f"   Consumers: {len(consumers)}")
            
            for consumer in consumers:
                print(f"     👤 {consumer.name}:")
                print(f"        Delivered: {consumer.delivered.stream_seq}")
                print(f"        Ack Floor: {consumer.ack_floor.stream_seq}")
                print(f"        Pending: {consumer.num_pending}")
                print(f"        Redelivered: {consumer.num_redelivered}")
        
        await nc.close()
        return True
        
    except Exception as e:
        print(f"❌ Error getting stream details: {e}")
        return False

def check_prometheus_targets():
    """Check if Prometheus is successfully scraping NATS exporter"""
    print("\n🎯 Prometheus Target Status:")
    print("-" * 40)
    
    try:
        response = requests.get("http://localhost:9090/api/v1/targets", timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            for target in data['data']['activeTargets']:
                job = target['labels']['job']
                health = target['health']
                instance = target['labels']['instance']
                last_scrape = target.get('lastScrape', 'N/A')
                
                symbol = "✅" if health == "up" else "❌"
                print(f"{symbol} {job} ({instance}): {health}")
                if health != "up":
                    print(f"    Error: {target.get('lastError', 'Unknown')}")
                else:
                    print(f"    Last Scrape: {last_scrape}")
            
            return True
        else:
            print(f"❌ Prometheus not reachable: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking Prometheus: {e}")
        return False

def check_recording_rules():
    """Check if Prometheus recording rules are working"""
    print("\n📏 Prometheus Recording Rules:")
    print("-" * 40)
    
    try:
        # Check if rules are loaded
        response = requests.get("http://localhost:9090/api/v1/rules", timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            rule_count = 0
            for group in data['data']['groups']:
                print(f"📁 Group: {group['name']}")
                for rule in group['rules']:
                    rule_count += 1
                    rule_type = rule['type']
                    name = rule['name'] if rule_type == 'recording' else rule['name']
                    health = rule.get('health', 'unknown')
                    
                    symbol = "✅" if health == "ok" else "❌" if health == "err" else "⚠️"
                    print(f"   {symbol} {rule_type}: {name} ({health})")
            
            print(f"\nTotal rules: {rule_count}")
            return True
        else:
            print(f"❌ Rules endpoint not reachable: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking rules: {e}")
        return False

async def main():
    print("🔍 JETSTREAM BACKLOG STATS & HEALTH CHECK")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Check all components
    metrics = check_prometheus_metrics()
    nats_info = check_nats_direct()
    await check_stream_details()
    targets_ok = check_prometheus_targets()
    rules_ok = check_recording_rules()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 HEALTH SUMMARY")
    print("=" * 60)
    
    checks = [
        ("NATS JetStream", nats_info.get('streams', 0) > 0),
        ("Prometheus Exporter", metrics.get('total_streams', 0) > 0),
        ("Prometheus Targets", targets_ok),
        ("Recording Rules", rules_ok),
        ("Consumers Active", metrics.get('total_consumers', 0) > 0)
    ]
    
    all_good = True
    for check_name, status in checks:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {check_name}")
        if not status:
            all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("✅ ALL SYSTEMS OPERATIONAL - READY FOR DRAIN TESTING")
    else:
        print("❌ SOME ISSUES DETECTED - REVIEW BEFORE PROCEEDING")
    
    return all_good

if __name__ == "__main__":
    asyncio.run(main())
