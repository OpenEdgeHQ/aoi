"""Versioned local mitigation protocol using live ITBench alert conditions.

This implements a disclosed research evaluation, not an upstream official verdict.
Pending and firing alerts both represent an active condition. Configuration
restoration and submit() are never used to infer mitigation success.
"""

import json
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen


PROTOCOL = "itbench_alert_resolution_v1"


def request(origin, path):
    with urlopen(origin.rstrip("/") + path, timeout=20) as response:
        payload = json.load(response)
    if payload.get("status") != "success":
        raise ValueError("Prometheus returned a non-success response")
    return payload["data"]


def namespace_matches(labels, namespaces):
    return any(labels.get(key) in namespaces for key in (
        "namespace", "exported_namespace", "source_workload_namespace", "destination_workload_namespace"))


def sample(origin, namespaces, expected):
    alerts = request(origin, "/api/v1/alerts")["alerts"]
    matching = [alert for alert in alerts if namespace_matches(alert["labels"], namespaces)
                and any(alert["labels"].get("alertname") == spec["name"]
                        and all(alert["labels"].get(k) == v for k, v in spec.get("labels", {}).items())
                        for spec in expected)
                and alert.get("state") in {"pending", "firing"}]
    groups = request(origin, "/api/v1/rules?type=alert")["groups"]
    rules = [rule for group in groups for rule in group["rules"]
             if rule.get("name") in {item["name"] for item in expected}]
    now = datetime.now(timezone.utc)
    rule_healthy = bool(expected) and all(any(rule.get("name") == spec["name"] for rule in rules) for spec in expected)
    for rule in rules:
        evaluated = datetime.fromisoformat(rule["lastEvaluation"].replace("Z", "+00:00"))
        rule_healthy = rule_healthy and rule.get("health") == "ok" and not rule.get("lastError") and 0 <= (now - evaluated).total_seconds() <= 90
    traffic = {}
    for namespace in namespaces:
        metric = "traces_span_metrics_calls_total" if namespace == "otel-demo" else "request_result_total"
        query = f'sum(rate({metric}{{namespace="{namespace}"}}[2m]))'
        result = request(origin, "/api/v1/query?" + urlencode({"query": query}))["result"]
        traffic[namespace] = bool(result) and float(result[0]["value"][1]) > 0
    return {"checked_at": now.isoformat(), "alerts": matching,
            "rules": [{key: rule.get(key) for key in ("name", "query", "duration", "labels", "health", "lastError", "lastEvaluation")} for rule in rules],
            "rule_healthy": bool(rule_healthy), "traffic_present": traffic,
            "telemetry_healthy": bool(rule_healthy) and all(traffic.values())}


def rule_signature(observation):
    return sorted(json.dumps({k: r.get(k) for k in ("name", "query", "duration", "labels")}, sort_keys=True)
                  for r in observation.get("rules", []))


def workload_health(baseline, final):
    checks = []
    for namespace, data in baseline.items():
        after = {(obj["kind"], obj["metadata"]["name"]): obj for obj in final.get(namespace, {}).get("items", [])}
        for obj in data.get("items", []):
            if obj["kind"] not in {"Deployment", "StatefulSet"} or obj["spec"].get("replicas", 1) < 1:
                continue
            current = after.get((obj["kind"], obj["metadata"]["name"]))
            healthy = False
            if current:
                desired = current["spec"].get("replicas", 1)
                status = current.get("status", {})
                healthy = (desired >= 1 and status.get("readyReplicas", 0) >= desired
                           and status.get("updatedReplicas", 0) >= desired
                           and status.get("observedGeneration", 0) >= current["metadata"].get("generation", 1))
            checks.append({"namespace": namespace, "kind": obj["kind"], "name": obj["metadata"]["name"], "healthy": healthy})
    return checks


def verdict(baseline_alerts, injected_alerts, post_samples, baseline_state, final_state):
    result = {"protocol": PROTOCOL, "source": "local_live_alert_evaluation", "official": False,
              "official_success": None, "passed": None, "reason": None}
    if not baseline_alerts.get("telemetry_healthy") or baseline_alerts.get("alerts"):
        result["reason"] = "baseline_not_healthy"
    elif not injected_alerts.get("rule_healthy") or not injected_alerts.get("alerts"):
        result["reason"] = "fault_alert_not_observed"
    elif len(post_samples) < 3 or any(not s.get("telemetry_healthy") for s in post_samples):
        result["reason"] = "post_evaluation_evidence_incomplete"
    elif any(rule_signature(s) != rule_signature(baseline_alerts) for s in [injected_alerts, *post_samples]):
        result["reason"] = "alert_rules_changed"
    else:
        checks = workload_health(baseline_state, final_state)
        result["workload_checks"] = checks
        result["passed"] = (all(not s["alerts"] for s in post_samples)
                            and bool(checks) and all(c["healthy"] for c in checks))
        result["reason"] = "alerts_cleared_and_workloads_healthy" if result["passed"] else "alerts_or_workload_fault_persist"
    return result
