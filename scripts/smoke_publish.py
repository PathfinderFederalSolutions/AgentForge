#!/usr/bin/env python3
"""
Smoke test for NATS publishing
"""
import asyncio
import sys
import os

try:
    import nats
except ImportError:
    print("nats-py not installed, skipping smoke test")
    sys.exit(0)


async def main():
    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    
    try:
        nc = await nats.connect(nats_url)
        print(f"✅ Connected to NATS at {nats_url}")
        
        # Publish a test message
        await nc.publish("test.smoke", b"smoke test message")
        print("✅ Published smoke test message")
        
        await nc.close()
        print("✅ Smoke test passed")
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
