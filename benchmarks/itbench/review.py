"""Compare saved Kubernetes configurations; this is not an ITBench verdict."""

import argparse
import json
from pathlib import Path


def objects(snapshot):
    return {(obj["kind"], obj["metadata"].get("namespace", ""), obj["metadata"]["name"]): obj
            for namespace in snapshot.values() for obj in namespace.get("items", [])
            if obj["kind"] != "Pod"}


def fields(value, path=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(fields(child, f"{path}/{key}"))
        return result
    if isinstance(value, list) and value and all(isinstance(item, dict) and "name" in item for item in value):
        result = {}
        for item in value:
            result.update(fields(item, f"{path}/name={item['name']}"))
        return result
    return {path: value}


def configuration(obj):
    if obj is None:
        return {"/exists": False}
    result = {"/exists": True, **fields(obj.get("spec", {}), "/spec")}
    if obj["kind"] == "Namespace":
        result.update(fields(obj["metadata"].get("labels", {}), "/metadata/labels"))
    return result


def review(baseline, injected, after):
    before, fault, final = map(objects, (baseline, injected, after))
    missing = object()
    changes = []
    for key in sorted(before.keys() | fault.keys()):
        b, i, a = (configuration(source.get(key)) for source in (before, fault, final))
        for path in sorted(b.keys() | i.keys()):
            if b.get(path, missing) != i.get(path, missing):
                changes.append({"kind": key[0], "namespace": key[1], "name": key[2],
                                "field": path, "matches_baseline_after": a.get(path, missing) == b.get(path, missing)})
    workloads = []
    for key, obj in sorted(final.items()):
        if obj["kind"] not in {"Deployment", "StatefulSet"}:
            continue
        desired = obj["spec"].get("replicas", 1)
        status = obj.get("status", {})
        workloads.append({"kind": key[0], "namespace": key[1], "name": key[2],
                          "desired": desired, "ready": status.get("readyReplicas", 0),
                          "generation_observed": status.get("observedGeneration", 0) >= obj["metadata"].get("generation", 1)})
    return {"source": "local_snapshot_review", "official": False, "official_success": None,
            "changed_fields": changes,
            "all_changed_fields_match_baseline": all(c["matches_baseline_after"] for c in changes) if changes else None,
            "workloads": workloads,
            "limitations": "Only captured configuration deltas and workload status; alternative valid repairs, traffic health, uncaptured resources and secrets are not graded."}


def review_directory(directory):
    snapshots = [json.loads((directory / name).read_text()) for name in ("baseline.json", "injected.json", "post-model.json")]
    if any(any("error" in namespace for namespace in snapshot.values()) for snapshot in snapshots):
        raise ValueError("Cannot review incomplete snapshots")
    result = review(*snapshots)
    (directory / "local-review.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(review_directory(args.scenario_directory), indent=2))


if __name__ == "__main__":
    main()
