"""Frozen child-process dispatch must precede every GUI import."""
import multiprocessing
import os
from pathlib import Path
import sys


if __name__ == '__main__':
    for stream in ('stdout', 'stderr'):
        if getattr(sys, stream) is None:
            setattr(sys, stream, open(os.devnull, 'w', encoding='utf-8'))
    multiprocessing.freeze_support()
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'backend'))
    from native.main import main
    raise SystemExit(main())
