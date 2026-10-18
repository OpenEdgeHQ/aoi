"""Durable per-event journal; survives runner termination without AOI changes."""

import json
import os
from datetime import datetime, timezone


def append_event(path, kind, value):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"recorded_at": datetime.now(timezone.utc).isoformat(), "kind": kind, "value": value}, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
