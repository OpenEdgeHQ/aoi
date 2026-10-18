"""Run AOI against one ITBench SRE mitigation scenario.

ITBench deploys/injects the scenario and owns the pass/fail verdict. AOI uses
the restricted kubeconfig for tool calls and writes an auditable task record.
"""

import argparse
import asyncio
import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

import llm_config
from benchmarks.itbench.client import ITBenchEnvironmentClient, load_itbench_verdict
from benchmarks.itbench.verify import verify
from benchmarks.itbench.preflight import require_permissions
from benchmarks.itbench.pricing import add_pricing_arguments, estimate_cost, validate_prices
from main import AIOPlatform
from utils.aworld_usage_capture import capture_aworld_usage
from benchmarks.itbench.journal import append_event

DEFAULT_MODEL = "gpt-4o-mini"


def _configure_model(model: str, temperature: float | None):
    if llm_config.API_SOURCE != "openai":
        return llm_config.create_llm_config(model=model, temperature=temperature)

    # Keep OpenAI compatibility local to this benchmark runner. The base
    # branch does not include the separate AIOpsLab model configuration fixes.
    from aworld.models.openai_provider import OpenAIProvider

    original = OpenAIProvider.get_openai_params
    if getattr(original, "_itbench_compat", False) is not True:
        def get_openai_params(self, *args, **kwargs):
            params = original(self, *args, **kwargs)
            return {key: value for key, value in params.items() if value is not None}

        get_openai_params._itbench_compat = True
        OpenAIProvider.get_openai_params = get_openai_params

    return llm_config.create_llm_config(
        model=model, temperature=temperature, disable_thinking=False,
    )


def _git_revision(directory: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _run_scenario_target(itbench_dir: Path, scenario_number: int, target: str) -> None:
    scenario_dir = itbench_dir / "scenarios" / "sre"
    if not (scenario_dir / "Makefile").is_file():
        raise FileNotFoundError(f"ITBench SRE Makefile not found: {scenario_dir}")
    subprocess.run(
        ["make", f"SCENARIO_NUMBER={scenario_number}", target],
        cwd=scenario_dir, check=True,
    )


def _token_usage(platform: AIOPlatform) -> dict:
    usage = {}
    for name in ("observer", "probe", "executor", "compressor"):
        agent = getattr(platform, name)
        if agent and hasattr(agent, "get_token_usage"):
            usage[name] = agent.get_token_usage()
        file_reader = getattr(agent, "file_reader", None) if agent else None
        if file_reader and hasattr(file_reader, "get_token_usage"):
            usage[f"{name}_file_reader"] = file_reader.get_token_usage()
    usage["total"] = {
        key: sum(item.get(key, 0) for item in usage.values())
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    return usage


async def run(args: argparse.Namespace) -> dict:
    validate_prices(args.input_usd_per_million, args.output_usd_per_million)
    if args.scenario_number < 1:
        raise ValueError("scenario-number must be positive")
    scenario_id = f"itbench-sre-{args.scenario_number}-mitigation-1"
    itbench_dir = args.itbench_dir.expanduser().resolve()
    kubeconfig = args.kubeconfig.expanduser().resolve()
    if args.start_scenario:
        _run_scenario_target(itbench_dir, args.scenario_number, "start-scenario")
        _run_scenario_target(itbench_dir, args.scenario_number, "enable-agent-access")

    client = ITBenchEnvironmentClient(
        scenario_id, args.incident, kubeconfig, args.prometheus_url,
        namespaces=getattr(args, "namespaces", None),
    )
    require_permissions(itbench_dir, args.scenario_number, kubeconfig, client.namespaces,
                        args.output.with_suffix(".preflight.json"))
    client.journal_path = args.output.parent / "events.jsonl"
    model = args.model or DEFAULT_MODEL
    config = _configure_model(model, args.temperature)
    platform = AIOPlatform(
        llm_config=config, env_client=client,
        max_iterations=args.max_iterations,
        max_context_tokens=args.max_context_tokens,
        max_output_tokens=args.max_output_tokens,
    )
    started_at = datetime.now(timezone.utc)
    start_clock = monotonic()
    attempt = None
    error = None
    append_event(client.journal_path, "trial_started", {
        "scenario_number": args.scenario_number, "started_at": started_at.isoformat(),
        "model": model, "temperature": config.llm_config.llm_temperature,
        "max_iterations": args.max_iterations, "max_context_tokens": args.max_context_tokens,
        "max_output_tokens": args.max_output_tokens,
        "input_usd_per_million": args.input_usd_per_million, "output_usd_per_million": args.output_usd_per_million})
    with capture_aworld_usage(on_usage=lambda usage: append_event(client.journal_path, "provider_usage", usage)) as provider_calls:
        try:
            attempt = await platform.run(problem_id=scenario_id)
        except (Exception, asyncio.CancelledError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    ended_at = datetime.now(timezone.utc)
    verdict = load_itbench_verdict(args.verdict_json)
    usage = _token_usage(platform)
    if provider_calls:
        usage = {}
        for call in provider_calls:
            bucket = usage.setdefault(call.get("role", "unattributed"), {
                key: 0 for key in ("input_tokens", "output_tokens", "total_tokens")
            })
            for key in bucket:
                bucket[key] += call[key]
        usage["total"] = {
            key: sum(call[key] for call in provider_calls)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        usage["source"] = "provider_reported"
    else:
        usage["source"] = "agent_reported" if usage["total"]["total_tokens"] else "unavailable"
    cost = estimate_cost(usage, args.input_usd_per_million, args.output_usd_per_million)
    raw_trace = Path("res/itbench/raw") / model / f"{client.session_id}.json"
    patch_result = subprocess.run(["git", "-C", str(itbench_dir), "diff", "HEAD"],
                                  capture_output=True, check=True)
    local_verification = None
    if getattr(args, "local_verify", False):
        try:
            local_verification = verify(kubeconfig, args.scenario_number)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            local_verification = {"official": False, "source": "local_scenario_check",
                                  "passed": None, "error": str(exc)}
    record = {
        "benchmark": "ITBench", "domain": "SRE", "task_type": "mitigation",
        "scenario_number": args.scenario_number, "task_id": scenario_id,
        "incident": args.incident,
        "itbench_revision": _git_revision(itbench_dir),
        "itbench_working_tree_dirty": bool(patch_result.stdout),
        "itbench_patch_sha256": hashlib.sha256(patch_result.stdout).hexdigest() if patch_result.stdout else None,
        "aoi_revision": _git_revision(Path.cwd()),
        "started_at": started_at.isoformat(), "ended_at": ended_at.isoformat(),
        "duration_seconds": round(monotonic() - start_clock, 3),
        "configuration": {
            **llm_config.backend_identity(),
            "namespaces": client.namespaces,
            "model": model, "temperature": config.llm_config.llm_temperature,
            "max_iterations": args.max_iterations,
            "max_context_tokens": args.max_context_tokens,
            "max_output_tokens": args.max_output_tokens,
            "scenario_started_by_runner": args.start_scenario,
            "kubeconfig_path": str(kubeconfig),
            "prometheus_url": args.prometheus_url,
            "input_usd_per_million": args.input_usd_per_million,
            "output_usd_per_million": args.output_usd_per_million,
        },
        "attempt": attempt, "submitted": client.submitted,
        "raw_aoi_trace": str(raw_trace) if raw_trace.is_file() else None,
        "verdict": verdict, "success": (
            verdict.get("success", verdict.get("passed")) if verdict is not None else None
        ),
        "local_verification": local_verification,
        "tool_calls": client.tool_calls, "token_usage": usage,
        "provider_usage_calls": provider_calls,
        "cost_usd": cost,
        "cost_note": (
            "Estimate at supplied standard rates; cached-input discounts are not applied." if cost is not None else
            "Model usage unavailable; cost cannot be estimated."
            if usage["source"] == "unavailable" else
            "Supply model input/output prices to estimate cost."
        ),
        "error": error,
        "record_complete": error is None,
        "usage_complete": error is None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(args.output)
    append_event(client.journal_path, "trial_ended", {"ended_at": ended_at.isoformat(), "error": error})
    if args.stop_scenario:
        _run_scenario_target(itbench_dir, args.scenario_number, "disable-agent-access")
        _run_scenario_target(itbench_dir, args.scenario_number, "stop-scenario")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AOI on one ITBench SRE mitigation scenario")
    parser.add_argument("--itbench-dir", type=Path, required=True)
    parser.add_argument("--scenario-number", type=int, required=True)
    parser.add_argument("--incident", required=True, help="Agent-visible incident text from ITBench")
    parser.add_argument("--namespace", dest="namespaces", action="append",
                        help="Scenario namespace metadata; repeat for multiple namespaces")
    parser.add_argument("--kubeconfig", type=Path, required=True, help="ITBench restricted agent kubeconfig")
    parser.add_argument("--prometheus-url", help="ITBench Prometheus base URL for alerts and PromQL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-iterations", type=int, default=14)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=5000)
    parser.add_argument("--start-scenario", action="store_true")
    parser.add_argument("--stop-scenario", action="store_true")
    parser.add_argument("--verdict-json", type=Path, help="JSON from ITBench grader with boolean success/passed")
    parser.add_argument("--local-verify", action="store_true", help="Check archived scenario 1 locally; never sets official success")
    add_pricing_arguments(parser)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("res/itbench") / f"sre-{args.scenario_number}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
    record = asyncio.run(run(args))
    print(f"ITBench task record: {args.output}")
    print(f"Submitted: {record['submitted']}; official success: {record['success']}")


if __name__ == "__main__":
    main()
