#!/usr/bin/env python3
"""Shared GPU-admission gate for the direct-PTX vision competitor runners.

A single definition keeps the admission rule identical across every vision
competitor so one evidence set cannot mix two different foreign-compute
thresholds. Runners import it directly; because Python places a script's own
directory on ``sys.path[0]``, this sibling import resolves regardless of the
caller's working directory.
"""

from __future__ import annotations

import os
import subprocess


def require_no_foreign_compute(label: str) -> None:
    """Fail closed when another process is using the GPU's compute engine."""
    monitor = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
                             check=True, capture_output=True, text=True, timeout=5)
    conflicts: list[str] = []
    for line in monitor.stdout.splitlines():
        cells = line.split()
        if not cells or cells[0].startswith("#") or len(cells) < 9:
            continue
        try:
            pid = int(cells[1])
        except ValueError:
            continue
        try:
            sm = int(cells[3])
        except ValueError:
            sm = 0
        process_type = cells[2].upper()
        if pid != os.getpid() and (process_type == "C" or
                                   ("C" in process_type and sm > 5)):
            conflicts.append(f"pid={pid} {cells[-1]} type={process_type} sm={sm}%")
    if conflicts:
        raise RuntimeError(f"[{label}] foreign GPU workload: " + "; ".join(conflicts))
