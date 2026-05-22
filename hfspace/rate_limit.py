"""Per-IP rate limiting for the /chat endpoint."""

import time
from collections import defaultdict

MAX_REQUESTS_PER_IP = 20
WINDOW_SECONDS = 60
MAX_GLOBAL_PER_MINUTE = 100

_ip_counters = defaultdict(list)
_global_counters = []


def is_rate_limited(client_ip):
    now = time.time()
    window_start = now - WINDOW_SECONDS

    _ip_counters[client_ip] = [
        t for t in _ip_counters[client_ip] if t > window_start
    ]

    if len(_ip_counters[client_ip]) >= MAX_REQUESTS_PER_IP:
        return True

    global _global_counters
    _global_counters = [t for t in _global_counters if t > window_start]
    if len(_global_counters) >= MAX_GLOBAL_PER_MINUTE:
        return True

    _ip_counters[client_ip].append(now)
    _global_counters.append(now)

    return False
