# Minimal stub used when USE_NATS_STUBS=1 to avoid live connections
import asyncio

class _FakeJS:
    async def publish(self, subject, payload=b"", headers=None):
        await asyncio.sleep(0)  # yield

class _FakeNC:
    async def subscribe(self, subject, queue=None, cb=None, inbox=None):  # returns a token
        # Immediately schedule a no-op to avoid hangs
        async def _noop(): 
            await asyncio.sleep(0)
        asyncio.create_task(_noop())
        return object()
    
    async def unsubscribe(self, token): 
        pass
    
    async def drain(self): 
        pass

async def get_nc_and_js():
    return _FakeNC(), _FakeJS()
