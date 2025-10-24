from .amem_backend import AMemBackend
from .mem0_backend import Mem0Backend
from .memos_backend import MemosBackend
from .memoryos_backend import MemoryOSBackend

BACKENDS = {
    "a-mem": AMemBackend,
    "mem0": Mem0Backend,
    "memos": MemosBackend,
    "memoryos": MemoryOSBackend
}

def build_memory(name: str, namespace: str = "ehr"):
    name = name.lower()
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name}")
    return BACKENDS[name](namespace=namespace)
