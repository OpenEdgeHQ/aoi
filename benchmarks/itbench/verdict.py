"""Attach a verdict produced by ITBench to a saved AOI task record."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.itbench.client import load_itbench_verdict


def attach_verdict(record_path: Path, verdict_path: Path) -> dict:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("benchmark") != "ITBench":
        raise ValueError("Record is not an ITBench attempt")
    verdict = load_itbench_verdict(verdict_path)
    record["verdict"] = verdict
    record["success"] = verdict.get("success", verdict.get("passed"))
    record["verdict_source"] = str(verdict_path.resolve())
    record["verdict_attached_at"] = datetime.now(timezone.utc).isoformat()
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach an external ITBench verdict")
    parser.add_argument("record", type=Path)
    parser.add_argument("verdict", type=Path)
    args = parser.parse_args()
    record = attach_verdict(args.record, args.verdict)
    print(f"{record['task_id']}: success={record['success']}")


if __name__ == "__main__":
    main()
