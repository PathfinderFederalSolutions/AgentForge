from swarm.memory.crdt import LWWMap
from swarm.memory.mesh import MemoryMesh

def test_crdt_merge_and_get():
    a = LWWMap()
    b = LWWMap()
    a.set("ns:k", 1, "A")
    b.set("ns:k", 2, "B")
    a.merge(b)
    assert a.get("ns:k") in (1, 2)

def test_memory_mesh_scope():
    m = MemoryMesh(scope="job:123", actor="agentX")
    m.set("status", "working")
    assert m.get("status") == "working"