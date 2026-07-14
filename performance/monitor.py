"""
Monitor de performance em tempo real: latência, eventos/s, uso de CPU/memória.
"""
import time
import psutil
import collections
from threading import Lock

class PerformanceMonitor:
    def __init__(self):
        self.lock = Lock()
        self.event_counts = collections.defaultdict(int)
        self.processing_times = collections.deque(maxlen=100)
        self.start_time = time.time()
        self.last_report = time.time()
    
    def record_event(self, event_type: str, processing_time: float = 0.0):
        with self.lock:
            self.event_counts[event_type] += 1
            if processing_time > 0:
                self.processing_times.append(processing_time)
    
    def snapshot(self) -> dict:
        with self.lock:
            now = time.time()
            elapsed = now - self.start_time
            events_per_sec = sum(self.event_counts.values()) / elapsed if elapsed > 0 else 0
            avg_latency = (sum(self.processing_times) / len(self.processing_times)
                           if self.processing_times else 0.0)
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            return {
                "events_per_sec": events_per_sec,
                "avg_latency_ms": avg_latency * 1000,
                "cpu_percent": cpu,
                "memory_percent": mem,
                "uptime_sec": elapsed,
                "event_counts": dict(self.event_counts)
            }