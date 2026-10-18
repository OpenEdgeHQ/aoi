"""Recover explicitly partial trial records from the durable event journal."""

import json
from datetime import datetime, timezone


def _tool_key(call):
    # Malformed model actions can be JSON lists/objects. Preserve them in the
    # journal while using a hashable identity to match starts with completions.
    return (call["timestamp"], call["role"], json.dumps(call["action"], sort_keys=True))


def recover_trial(directory, scenario_number, configuration, error):
    events = []
    journal = directory / "events.jsonl"
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A killed process may leave the final line incomplete.
    usage_calls = [e["value"] for e in events if e["kind"] == "provider_usage"]
    tools = [e["value"] for e in events if e["kind"] == "tool_call"]
    starts = [e["value"] for e in events if e["kind"] == "trial_started"]
    totals = {key: sum(c[key] for c in usage_calls) for key in ("input_tokens", "output_tokens", "total_tokens")}
    cost = None
    if usage_calls and configuration.get("input_usd_per_million") is not None:
        cost = round((totals["input_tokens"] * configuration["input_usd_per_million"]
                      + totals["output_tokens"] * configuration["output_usd_per_million"]) / 1e6, 6)
    started_at = starts[0]["started_at"] if starts else None
    ended_at = datetime.now(timezone.utc).isoformat()
    completed_keys = {_tool_key(c) for c in tools}
    in_flight = [e["value"] for e in events if e["kind"] == "tool_action_started"
                 and _tool_key(e["value"]) not in completed_keys]
    return {"benchmark": "ITBench", "domain": "SRE", "task_type": "mitigation",
            "scenario_number": scenario_number, "configuration": configuration,
            "started_at": started_at, "ended_at": ended_at,
            "duration_seconds": None, "timing_note": "Wall timestamps only; process monotonic duration unavailable",
            "attempt": None, "submitted": any(c["result"].get("is_submission") for c in tools),
            "success": None, "verdict": None, "tool_calls": tools, "in_flight_tool_calls": in_flight,
            "provider_usage_calls": usage_calls,
            "token_usage": {"source": "provider_reported_partial" if usage_calls else "unavailable", "total": totals if usage_calls else None},
            "cost_usd": cost, "cost_note": "Partial completed-provider-call estimate; in-flight usage unknown",
            "record_complete": False, "usage_complete": False, "error": error}
