"""Check known repair permissions without exposing fault metadata to the model.

This is a permission check, not a guarantee that every repair strategy can run.
Unknown fault types are reported explicitly; no privileges are granted here.
"""

import json
import subprocess
from pathlib import Path

import yaml

ENV_FAULT = "modified-kubernetes-workload-container-environment-variable"
PVC_FAULT = "nonexistent-kubernetes-workload-persistent-volume-claim"
NAMESPACE_FAULT = "disabled-istio-ambient-mode-kubernetes-namespace"


def required_permissions(itbench_dir: Path, scenario_number: int, namespaces: list[str]) -> tuple[list[dict], list[str]]:
    source = Path(itbench_dir) / f"archive/scenarios/{scenario_number}/scenario.yaml"
    scenario = yaml.safe_load(source.read_text())
    checks = [{"verb": "list", "resource": resource, "namespace": ns}
              for ns in namespaces for resource in ("pods", "deployments.apps")]
    unchecked = []
    for group in scenario.get("faults", []):
        for injection in group.get("injections", []):
            fault = injection["id"]
            obj = injection.get("args", {}).get("kubernetesObject", {})
            meta = obj.get("metadata", {})
            if fault == NAMESPACE_FAULT and obj.get("kind") == "Namespace":
                checks.append({"verb": "patch", "resource": f"namespaces/{meta['name']}"})
            elif fault in {ENV_FAULT, PVC_FAULT} and obj.get("kind") == "Deployment":
                ns = meta["namespace"]
                checks.append({"verb": "patch", "resource": f"deployments.apps/{meta['name']}", "namespace": ns})
                if fault == PVC_FAULT:
                    checks.extend({"verb": verb, "resource": "persistentvolumeclaims", "namespace": ns}
                                  for verb in ("get", "create", "delete"))
            else:
                unchecked.append(fault)
    return [dict(items) for items in dict.fromkeys(tuple(c.items()) for c in checks)], sorted(set(unchecked))


def check_permissions(itbench_dir: Path, scenario_number: int, kubeconfig: Path,
                      namespaces: list[str]) -> dict:
    checks, unchecked = required_permissions(itbench_dir, scenario_number, namespaces)
    results = []
    for check in checks:
        argv = ["kubectl", "--kubeconfig", str(kubeconfig), "auth", "can-i", check["verb"], check["resource"]]
        if check.get("namespace"):
            argv.extend(["--namespace", check["namespace"]])
        try:
            process = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
            answer = process.stdout.strip()
            allowed = process.returncode == 0 and answer == "yes"
            detail = None if allowed else (process.stderr.strip() or answer or f"kubectl exited {process.returncode}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            allowed, detail = False, str(exc)
        results.append({**check, "allowed": allowed, "detail": detail})
    return {"scenario_number": scenario_number, "model_executed": False,
            "passed": all(c["allowed"] for c in results), "checks": results,
            "unchecked_fault_ids": unchecked,
            "coverage": "partial" if unchecked else "known_repair_operations"}


def require_permissions(itbench_dir: Path, scenario_number: int, kubeconfig: Path,
                        namespaces: list[str], output: Path) -> dict:
    report = check_permissions(itbench_dir, scenario_number, kubeconfig, namespaces)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        denied = [f"{c['verb']} {c['resource']}" + (f" in {c['namespace']}" if c.get('namespace') else '')
                  for c in report["checks"] if not c["allowed"]]
        raise ValueError("ITBench permission preflight blocked; model trial not started: "
                         + "; ".join(denied) + f". See {output}")
    return report
