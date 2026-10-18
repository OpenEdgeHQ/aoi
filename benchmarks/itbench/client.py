"""Local ITBench SRE environment adapter for the existing AOI harness.

ITBench owns scenario deployment and grading. This client connects AOI to an
already started scenario using ITBench's restricted kubeconfig. Submission is
recorded, never interpreted as a passing benchmark verdict.
"""

import ast
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen
from benchmarks.itbench.journal import append_event


READ_ONLY_VERBS = {
    "get", "describe", "logs", "top", "api-resources", "api-versions",
    "cluster-info", "version", "explain",
}
GLOBAL_OPTIONS_WITH_VALUES = {"-n", "--namespace", "--request-timeout"}
FORBIDDEN_CONNECTION_OPTIONS = {
    "-s", "--as", "--as-group", "--as-uid", "--insecure-skip-tls-verify",
    "--kubeconfig", "--context", "--server", "--user", "--cluster",
    "--token", "--client-certificate", "--client-key", "--certificate-authority",
}


def _kubectl_verb(args: list[str]) -> str:
    index = 0
    while index < len(args):
        token = args[index]
        if token in GLOBAL_OPTIONS_WITH_VALUES:
            index += 2
        elif token.startswith("-"):
            index += 1
        else:
            return token
    return ""


def _uses_stdin_manifest(argv: list[str]) -> bool:
    """Require one stdin filename, independent of other kubectl option positions."""
    filenames = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in {"-f", "--filename"}:
            index += 1
            if index == len(argv):
                return False
            filenames.append(argv[index])
        elif arg.startswith("--filename="):
            filenames.append(arg.split("=", 1)[1])
        elif arg.startswith("-f") and len(arg) > 2:
            filenames.append(arg[2:].removeprefix("="))
        elif arg == "--kustomize" or arg.startswith("--kustomize=") or arg.startswith("-k"):
            return False
        index += 1
    return filenames == ["-"]


class ITBenchEnvironmentClient:
    benchmark_name = "ITBench"

    def __init__(self, scenario_id: str, incident: str, kubeconfig: Path,
                 prometheus_url: str | None = None, namespaces: list[str] | None = None):
        self.scenario_id = scenario_id
        self.incident = incident
        self.namespaces = list(dict.fromkeys(namespaces or []))
        self.kubeconfig = Path(kubeconfig).expanduser().resolve()
        if not self.kubeconfig.is_file():
            raise FileNotFoundError(f"ITBench kubeconfig not found: {self.kubeconfig}")
        if prometheus_url:
            parsed = urlparse(prometheus_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
                raise ValueError("prometheus_url must be an HTTP(S) origin without credentials")
            self.prometheus_url = prometheus_url.rstrip("/")
        else:
            self.prometheus_url = None
        self.session_id = f"itbench-{scenario_id}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
        self.submitted = False
        self.repaired = False
        self.verified_after_repair = False
        self.tool_calls: list[dict[str, Any]] = []
        self.journal_path = None

    def init_problem(self, problem_id: str) -> dict[str, Any]:
        if problem_id != self.scenario_id:
            raise ValueError(f"Expected ITBench scenario {self.scenario_id}, got {problem_id}")
        if self.namespaces:
            examples = "; ".join(f"exec_kubectl({['get', 'pods', '-n', namespace]!r})"
                                 for namespace in self.namespaces)
            namespace_instruction = f"Scenario namespaces: {', '.join(self.namespaces)}. Inspect with {examples}. "
        else:
            namespace_instruction = (
                "Use exec_kubectl(['get', 'pods', '-n', '<incident-namespace>']); "
                "replace <incident-namespace> with a namespace identified in the incident. "
                "No scenario namespace metadata was supplied; do not assume a default namespace. "
            )
        return {
            "session_id": self.session_id,
            "problem_id": problem_id,
            "namespaces": self.namespaces,
            "task_description": self.incident,
            "instructions": (
                namespace_instruction + "Pass each kubectl argument "
                "as a separate string; JSON patches are single arguments. For apply -f -, provide "
                "stdin='YAML or JSON'. Return the entire action as a string, not a list or object. "
                "Within a Python single-quoted stdin string, escape newlines as \\n; "
                "literal multiline content requires triple quotes. "
                "Never use kubectl edit, shell operators, or heredocs. "
                "The restricted account may not list namespaces; use the incident namespace explicitly. "
                "Inspect actual configuration and telemetry before choosing a repair. "
                "Use kubectl set env for individual environment changes and set resources for limits; "
                "do not invent resource names or infer a traffic cause solely from old pod events. "
                "After repair, read the affected configuration and check workload/metric recovery. "
                "If Prometheus is configured, use get_alerts() or query_prometheus('PromQL'). "
                "Probe may only inspect; Executor may change resources. "
                "Use submit() when finished. The external ITBench grader determines success."
            ),
            "available_actions": {
                "exec_kubectl": "Run kubectl with a list of arguments; optional stdin string for apply -f -",
                "exec_shell": "Run kubectl against ITBench's restricted kubeconfig",
                "get_alerts": "Read active Prometheus alerts when configured",
                "query_prometheus": "Run one read-only PromQL query when configured",
                "submit": "Finish this attempt; grading happens separately",
            },
        }

    def get_session_id(self) -> str:
        return self.session_id

    def get_submit_format(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "problem_id": self.scenario_id,
            "submit_api": {"api_name": "submit", "description": "Call submit() after repair and verification"},
            "is_already_submitted": self.submitted,
        }

    def execute_probe_action(self, action: str) -> str:
        result = self._execute(action, role="probe")
        return f"Error: {result['result']}" if result.get("error") else result["result"]

    def execute_action(self, action: str) -> dict[str, Any]:
        return self._execute(action, role="executor")

    def _execute(self, action: str, role: str) -> dict[str, Any]:
        started = datetime.now(timezone.utc)
        append_event(self.journal_path, "tool_action_started", {"timestamp": started.isoformat(), "role": role, "action": action})
        try:
            if not isinstance(action, str):
                raise ValueError("Action must be a Python call string, not a list/object; "
                                 "use exec_kubectl(['get', 'pods', '-n', '<incident-namespace>']) as a string")
            expression = ast.parse(action.strip(), mode="eval").body
            if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
                raise ValueError("Expected exec_shell('kubectl ...') or submit()")
            if expression.func.id == "submit":
                if expression.args or expression.keywords or role == "probe":
                    raise ValueError("submit() takes no arguments and is unavailable to Probe")
                self.submitted = True
                result = {"result": "submitted; external ITBench verdict pending", "is_submission": True}
            elif expression.func.id in {"get_alerts", "query_prometheus"}:
                if not self.prometheus_url:
                    raise ValueError("Prometheus URL was not configured")
                if expression.keywords:
                    raise ValueError("Keyword arguments are unsupported")
                if expression.func.id == "get_alerts":
                    if expression.args:
                        raise ValueError("get_alerts() takes no arguments")
                    endpoint = "/api/v1/alerts"
                else:
                    if len(expression.args) != 1:
                        raise ValueError("query_prometheus requires one PromQL string")
                    query = ast.literal_eval(expression.args[0])
                    if not isinstance(query, str):
                        raise ValueError("PromQL must be a string")
                    endpoint = "/api/v1/query?" + urlencode({"query": query})
                with urlopen(self.prometheus_url + endpoint, timeout=20) as response:
                    result = {"result": response.read(200_000).decode("utf-8", errors="replace")}
            elif expression.func.id in {"exec_shell", "exec_kubectl"}:
                stdin = None
                if len(expression.args) != 1:
                    raise ValueError("Provide one command string or one list of kubectl arguments")
                value = ast.literal_eval(expression.args[0])
                if expression.func.id == "exec_shell":
                    if not isinstance(value, str) or expression.keywords:
                        raise ValueError("exec_shell requires one string command")
                    argv = shlex.split(value)
                    if not argv or argv.pop(0) != "kubectl":
                        raise ValueError("Only kubectl commands are supported by this ITBench adapter")
                    if any(token in argv for token in (";", "|", "&&", "||", ">", "<")) or any("<<" in token for token in argv):
                        raise ValueError("Shell operators and heredocs are unsupported; use exec_kubectl with stdin")
                else:
                    if not isinstance(value, list) or not value or not all(isinstance(arg, str) for arg in value):
                        raise ValueError("exec_kubectl requires a nonempty list of string arguments")
                    argv = value
                    if expression.keywords:
                        if len(expression.keywords) != 1 or expression.keywords[0].arg != "stdin":
                            raise ValueError("Only the stdin keyword is supported")
                        stdin = ast.literal_eval(expression.keywords[0].value)
                        if not isinstance(stdin, str):
                            raise ValueError("stdin must be a string")
                if any(arg.split("=", 1)[0] in FORBIDDEN_CONNECTION_OPTIONS for arg in argv):
                    raise ValueError("Changing kubeconfig or connection settings is not allowed")
                verb = _kubectl_verb(argv)
                if verb in {"edit", "attach", "exec", "port-forward", "proxy"}:
                    raise ValueError(f"Interactive or remote shell verb {verb} is unsupported; use get, patch, apply, set, or rollout")
                if role == "probe" and verb not in READ_ONLY_VERBS:
                    raise ValueError(f"Probe cannot run state-changing kubectl verb: {verb}")
                if stdin is not None and (verb != "apply" or not _uses_stdin_manifest(argv)):
                    raise ValueError("stdin requires apply with exactly one stdin filename (-f - or --filename=-)")
                completed = subprocess.run(
                    ["kubectl", "--kubeconfig", str(self.kubeconfig), *argv],
                    input=stdin,
                    capture_output=True, text=True, timeout=90, check=False,
                )
                output = (completed.stdout + completed.stderr).strip()
                result = {"result": output or f"kubectl exited {completed.returncode}"}
                if completed.returncode:
                    result["error"] = True
                elif verb not in READ_ONLY_VERBS and not (verb == "rollout" and "status" in argv):
                    self.repaired = True
                    self.verified_after_repair = False
                elif self.repaired:
                    self.verified_after_repair = True
            else:
                raise ValueError(f"Unsupported action: {expression.func.id}")
        except (SyntaxError, ValueError, TypeError, subprocess.TimeoutExpired, OSError) as exc:
            message = str(exc)
            if isinstance(exc, SyntaxError):
                message += ("; provide a Python call string. For multiline stdin, escape newlines as "
                            "\\n or use a triple-quoted Python string; do not put literal newlines "
                            "inside a single-quoted string. Example: "
                            "exec_kubectl(['apply', '-f', '-'], stdin='kind: ConfigMap\\n')")
            result = {"result": message, "error": True}
        self.tool_calls.append({
            "timestamp": started.isoformat(), "role": role, "action": action,
            "duration_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            "result": result,
        })
        append_event(self.journal_path, "tool_call", self.tool_calls[-1])
        return result

    def is_problem_solved(self) -> bool:
        # Only ITBench's verifier can establish benchmark success.
        return False

    def cleanup_session(self) -> bool:
        # Scenario lifecycle belongs to ITBench, not this adapter.
        return True


def load_itbench_verdict(path: Path | None) -> dict[str, Any] | None:
    """Read an externally produced verdict without inferring success from AOI output."""
    if path is None:
        return None
    verdict = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(verdict, dict):
        raise ValueError("ITBench verdict must be a JSON object")
    if verdict.get("official") is False or verdict.get("source") == "local_scenario_check":
        raise ValueError("Local checks cannot be imported as an official ITBench verdict")
    success = verdict.get("success", verdict.get("passed"))
    if not isinstance(success, bool):
        raise ValueError("ITBench verdict needs a boolean success or passed field")
    return verdict
