"""Atomic replacement tolerates brief Windows sharing locks, then fails clearly."""
import os
import time


def replace_with_retry(source,target):
    for attempt in range(5):
        try:
            os.replace(source,target)
            return
        except PermissionError as error:
            if attempt==4:raise
            # Antivirus and indexers can hold a new file through the first
            # few hundred milliseconds. Keep permanent failures bounded.
            delay = 0.15 if getattr(error, 'winerror', None) in (5, 32, 33) else 0.03
            time.sleep(delay*(2**attempt))
