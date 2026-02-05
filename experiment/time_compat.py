"""
Compatibility patch for time.clock() which was removed in Python 3.8+
"""

import time

# time.clock() was removed in Python 3.8
# Replace it with time.perf_counter() for compatibility
if not hasattr(time, "clock"):
    time.clock = time.perf_counter
