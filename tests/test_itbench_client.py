import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.itbench.verdict import attach_verdict
from benchmarks.itbench.client import ITBenchEnvironmentClient, load_itbench_verdict
from benchmarks.itbench.summary import summarize
from benchmarks.itbench.run import DEFAULT_MODEL, _configure_model
from utils.aworld_usage_capture import capture_aworld_usage
from utils.llm_usage import usage_from_result


class ITBenchEnvironmentClientTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.kubeconfig = Path(self.directory.name) / "restricted-kubeconfig"
        self.kubeconfig.write_text("test", encoding="utf-8")
        self.client = ITBenchEnvironmentClient(
            "itbench-sre-1-mitigation-1", "Resolve an active alert", self.kubeconfig,
        )

    def tearDown(self):
        self.directory.cleanup()

    @patch("benchmarks.itbench.client.subprocess.run")
    def test_probe_uses_restricted_kubeconfig(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = "pod-a Running\n"
        run.return_value.stderr = ""
        result = self.client.execute_probe_action("exec_shell('kubectl get pods -n app')")
        self.assertIn("pod-a Running", result)
        self.assertEqual(
            run.call_args.args[0],
            ["kubectl", "--kubeconfig", str(self.kubeconfig.resolve()), "get", "pods", "-n", "app"],
        )

    @patch("benchmarks.itbench.client.subprocess.run")
    def test_probe_rejects_changes_and_context_override(self, run):
        self.assertIn("Probe cannot", self.client.execute_probe_action(
            "exec_shell('kubectl delete pod pod-a')"))
        self.assertIn("not allowed", self.client.execute_probe_action(
            "exec_shell('kubectl --context other get pods')"))
        run.assert_not_called()

    def test_submission_is_not_a_success_verdict(self):
        result = self.client.execute_action("submit()")
        self.assertTrue(result["is_submission"])
        self.assertTrue(self.client.submitted)
        self.assertFalse(self.client.is_problem_solved())

    @patch("benchmarks.itbench.client.subprocess.run")
    def test_patch_json_is_passed_without_shell_quoting(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = "patched"
        run.return_value.stderr = ""
        payload = json.dumps({"data": {"demo.flagd.json": '{"flags": {"name": "value with apostrophe\'s"}}'}})
        args = ['patch', 'configmap', 'flagd-config', '-n', 'otel-demo', '--type=merge', '-p', payload]
        result = self.client.execute_action(f"exec_kubectl({args!r})")
        self.assertNotIn("error", result)
        self.assertEqual(run.call_args.args[0][-1], payload)
        self.assertFalse(self.client.verified_after_repair)

    @patch("benchmarks.itbench.client.subprocess.run")
    def test_stdin_apply_is_noninteractive(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = "configured"
        run.return_value.stderr = ""
        manifest = 'apiVersion: v1\nkind: ConfigMap\n'
        self.client.execute_action(f"exec_kubectl(['apply', '-n', 'otel-demo', '-f', '-'], stdin={manifest!r})")
        self.assertEqual(run.call_args.kwargs['input'], manifest)

    @patch("benchmarks.itbench.client.subprocess.run")
    def test_interactive_commands_never_start(self, run):
        for verb in ('edit', 'exec', 'port-forward'):
            result = self.client.execute_action(f"exec_kubectl([{verb!r}, 'pod-a'])")
            self.assertTrue(result['error'])
        run.assert_not_called()

    @patch("benchmarks.itbench.client.subprocess.run")
    def test_submission_does_not_impose_extra_repair_or_read_gate(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = 'ok'
        run.return_value.stderr = ''
        self.assertTrue(self.client.execute_action('submit()')['is_submission'])
        self.client.execute_action("exec_kubectl(['patch', 'configmap', 'flags', '-p', '{}'])")
        self.assertTrue(self.client.execute_action('submit()')['is_submission'])
        self.client.execute_probe_action("exec_kubectl(['get', 'configmap', 'flags'])")
        self.assertTrue(self.client.execute_action('submit()')['is_submission'])
        self.assertFalse(self.client.is_problem_solved())

    @patch("benchmarks.itbench.client.subprocess.run")
    def test_failed_tools_do_not_change_submission_into_a_verdict(self, run):
        run.return_value.returncode = 1
        run.return_value.stdout = ''
        run.return_value.stderr = 'failed'
        self.client.execute_action("exec_kubectl(['patch', 'configmap', 'flags', '-p', '{}'])")
        self.assertFalse(self.client.repaired)
        self.client.repaired = True
        self.client.execute_probe_action("exec_kubectl(['get', 'configmap', 'flags'])")
        self.assertTrue(self.client.execute_action('submit()')['is_submission'])
        self.assertFalse(self.client.is_problem_solved())

    @patch("benchmarks.itbench.client.subprocess.run")
    def test_structured_probe_cannot_change_connection_or_resources(self, run):
        for args in (['delete', 'pod', 'x'], ['--as', 'admin', 'get', 'pods'], ['-s', 'other', 'get', 'pods']):
            self.assertIn('Error:', self.client.execute_probe_action(f'exec_kubectl({args!r})'))
        run.assert_not_called()

    @patch("benchmarks.itbench.client.urlopen")
    def test_prometheus_alerts_use_configured_endpoint(self, urlopen):
        client = ITBenchEnvironmentClient(
            "itbench-sre-1-mitigation-1", "Resolve alert", self.kubeconfig,
            "http://prometheus.example:9090/prometheus",
        )
        response = urlopen.return_value.__enter__.return_value
        response.read.return_value = b'{"status":"success"}'
        result = client.execute_probe_action("get_alerts()")
        self.assertIn('"success"', result)
        self.assertEqual(urlopen.call_args.args[0], "http://prometheus.example:9090/prometheus/api/v1/alerts")

    def test_only_external_boolean_verdict_counts(self):
        path = Path(self.directory.name) / "verdict.json"
        path.write_text(json.dumps({"passed": True}), encoding="utf-8")
        self.assertTrue(load_itbench_verdict(path)["passed"])
        path.write_text(json.dumps({"status": "submitted"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_itbench_verdict(path)
        path.write_text('{"official": false, "passed": true}', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_itbench_verdict(path)

    def test_attach_verdict_updates_saved_record(self):
        record_path = Path(self.directory.name) / "attempt.json"
        verdict_path = Path(self.directory.name) / "verdict.json"
        record_path.write_text(json.dumps({
            "benchmark": "ITBench", "task_id": "itbench-sre-1-mitigation-1",
            "success": None,
        }), encoding="utf-8")
        verdict_path.write_text('{"passed": false}', encoding="utf-8")
        record = attach_verdict(record_path, verdict_path)
        self.assertIs(record["success"], False)
        self.assertIs(json.loads(record_path.read_text())["success"], False)

    def test_token_usage_handles_aworld_result_shape(self):
        result = type("Result", (), {"usage": {
            "prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16,
        }})()
        self.assertEqual(usage_from_result(result), {
            "input_tokens": 12, "output_tokens": 4, "total_tokens": 16,
        })

    def test_provider_usage_capture(self):
        import aworld.models.openai_provider as provider_module
        with patch.object(provider_module, "usage_process") as original:
            with capture_aworld_usage() as calls:
                provider_module.usage_process({
                    "prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16,
                })
            self.assertEqual(calls[0]["total_tokens"], 16)
            original.assert_called_once()

    def test_provider_usage_has_agent_attribution_and_restores_runner(self):
        import asyncio
        import aworld.models.openai_provider as provider_module
        from aworld.runner import Runners
        async def fake_run(**kwargs):
            provider_module.usage_process({'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7})
        with patch.object(Runners, 'run', fake_run), patch.object(provider_module, 'usage_process'):
            with capture_aworld_usage() as calls:
                asyncio.run(Runners.run(agent=type('Agent', (), {'name': 'Probe Agent'})()))
            self.assertEqual(calls[0]['role'], 'probe')
            self.assertIs(Runners.run, fake_run)

    def test_openai_configuration_stays_with_itbench_runner(self):
        from aworld.models.openai_provider import OpenAIProvider

        with patch("llm_config.API_SOURCE", "openai"), patch.object(
            OpenAIProvider, "get_openai_params", return_value={
                "model": "gpt-4o-mini", "max_tokens": 100, "unused": None,
            },
        ):
            config = _configure_model(DEFAULT_MODEL, 0.7)
            params = OpenAIProvider.get_openai_params(None)

        self.assertEqual(DEFAULT_MODEL, "gpt-4o-mini")
        self.assertEqual(config.llm_config.llm_temperature, 0.7)
        self.assertNotIn("extra_body", getattr(config, "ext", {}))
        self.assertEqual(params, {"model": "gpt-4o-mini", "max_tokens": 100})

    def test_summary_excludes_unverified_attempts_from_success_rate(self):
        result = summarize([
            {"success": True, "duration_seconds": 10, "tool_calls": [{}], "cost_usd": 0.2},
            {"success": False, "duration_seconds": 20, "tool_calls": [{}, {}], "cost_usd": 0.4},
            {"success": None, "duration_seconds": 30, "tool_calls": []},
        ])
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["unverified_attempts"], 1)
        self.assertEqual(result["mean_duration_seconds"], 15)


if __name__ == "__main__":
    unittest.main()
