"""Atomic replacement tolerates brief Windows sharing locks, then fails clearly."""
import os
import time


def replace_with_retry(source,target):
    for attempt in range(5):
        try:
            os.replace(source,target)
            return
        except PermissionError:
            if attempt==4:raise
            time.sleep(0.03*(2**attempt))
