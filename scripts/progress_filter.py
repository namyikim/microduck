#!/usr/bin/env python3
"""Add a compact percentage progress bar to RSL-RL console output."""

from __future__ import annotations

import re
import sys

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ITER_RE = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")


def progress_bar(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return ""
    pct = max(0.0, min(100.0, current / total * 100.0))
    filled = min(width, int(round(width * pct / 100.0)))
    bar = "█" * filled + "░" * (width - filled)
    return f"[MicroDuck progress] [{bar}] {pct:5.1f}%  ({current:,}/{total:,})"


for raw_line in sys.stdin:
    sys.stdout.write(raw_line)
    sys.stdout.flush()

    clean = ANSI_RE.sub("", raw_line)
    match = ITER_RE.search(clean)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        # RSL-RL iteration indices are zero-based in a fresh run.
        display_current = min(total, current + 1)
        sys.stdout.write(progress_bar(display_current, total) + "\n")
        sys.stdout.flush()
