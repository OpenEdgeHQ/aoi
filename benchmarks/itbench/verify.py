"""Independent local checks for the archived SRE scenario 1; not official scoring."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def check_scenario_one(configmap: dict, deployments: dict) -> dict:
    data = json.loads(configmap["data"]["demo.flagd.json"])
    flag = data["flags"]["loadGeneratorFloodHomepage"]
    variant = flag.get("defaultVariant")
    disabled = flag.get("state") == "DISABLED" or flag.get("variants", {}).get(variant) == 0
    checks = [{"name": "homepage_flood_disabled", "passed": disabled,
               "state": flag.get("state"), "default_variant": variant}]
    for name in ("flagd", "load-generator"):
        deployment = deployments[name]
        desired = deployment["spec"].get("replicas", 1)
        status = deployment.get("status", {})
        ready = (desired >= 1 and status.get("readyReplicas", 0) == desired
                 and status.get("updatedReplicas", 0) == desired
                 and status.get("observedGeneration", 0) >= deployment["metadata"]["generation"])
        checks.append({"name": f"{name}_ready", "passed": ready, "desired_replicas": desired,
                       "ready_replicas": status.get("readyReplicas", 0)})
    return {"source": "local_scenario_check", "official": False,
            "scenario_number": 1, "passed": all(item["passed"] for item in checks),
            "checks": checks, "checked_at": datetime.now(timezone.utc).isoformat()}


def verify(kubeconfig: Path, scenario_number: int) -> dict:
    if scenario_number != 1:
        raise ValueError("Local verification is currently implemented only for archived scenario 1")
    def get(kind, name):
        result = subprocess.run(
            ["kubectl", "--kubeconfig", str(kubeconfig.resolve()), "get", kind, name,
             "-n", "otel-demo", "-o", "json"], capture_output=True, text=True,
            timeout=30, check=True,
        )
        return json.loads(result.stdout)
    return check_scenario_one(get("configmap", "flagd-config"), {
        name: get("deployment", name) for name in ("flagd", "load-generator")
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", required=True, type=Path)
    parser.add_argument("--scenario-number", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    result = verify(args.kubeconfig, args.scenario_number)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.record:
        record = json.loads(args.record.read_text())
        if record.get("benchmark") != "ITBench" or record.get("scenario_number") != args.scenario_number:
            raise ValueError("Record does not match the verified ITBench scenario")
        record["local_verification"] = result
        args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
