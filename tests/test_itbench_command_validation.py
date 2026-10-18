import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from benchmarks.itbench.client import ITBenchEnvironmentClient
from benchmarks.itbench.journal import append_event
from benchmarks.itbench.recover import recover_trial


class CommandValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        kubeconfig = Path(self.temp.name) / 'config'; kubeconfig.write_text('test')
        self.client = ITBenchEnvironmentClient('test', 'Investigate', kubeconfig)

    @patch('benchmarks.itbench.client.subprocess.run')
    def test_non_string_commands_are_logged_errors_and_can_retry(self, run):
        for action in (['exec_kubectl', ['get', 'pods']], {'command': 'get pods'}, None, 42):
            result = self.client.execute_action(action)
            self.assertTrue(result['error'])
            self.assertIn('Python call string', result['result'])
            self.assertEqual(self.client.tool_calls[-1]['action'], action)
        run.assert_not_called()
        run.return_value.returncode = 0; run.return_value.stdout = 'ok'; run.return_value.stderr = ''
        result = self.client.execute_action("exec_kubectl(['get', 'pods'])")
        self.assertNotIn('error', result)
        self.assertEqual(len(self.client.tool_calls), 5)

    @patch('benchmarks.itbench.client.subprocess.run')
    def test_probe_non_string_has_actionable_error(self, run):
        result = self.client.execute_probe_action(['exec_kubectl', ['get', 'pods']])
        self.assertIn('Error: Action must', result)
        self.assertTrue(self.client.tool_calls[-1]['result']['error'])
        run.assert_not_called()

    @patch('benchmarks.itbench.client.subprocess.run')
    def test_raw_multiline_string_is_rejected_with_specific_guidance(self, run):
        result = self.client.execute_action("exec_kubectl(['apply', '-f', '-'], stdin='kind: ConfigMap\nmetadata:\n  name: test\n')")
        self.assertTrue(result['error'])
        self.assertIn('triple-quoted', result['result'])
        self.assertIn('\\n', result['result'])
        self.assertTrue(self.client.tool_calls[-1]['result']['error'])
        run.assert_not_called()

    @patch('benchmarks.itbench.client.subprocess.run')
    def test_escaped_and_triple_quoted_multiline_stdin_are_identical(self, run):
        run.return_value.returncode = 0; run.return_value.stdout = 'configured'; run.return_value.stderr = ''
        manifest = 'kind: ConfigMap\nmetadata:\n  name: test\n'
        for literal in (repr(manifest), "'''" + manifest + "'''"):
            result = self.client.execute_action(f"exec_kubectl(['apply', '-f', '-'], stdin={literal})")
            self.assertNotIn('error', result)
            self.assertEqual(run.call_args.kwargs['input'], manifest)

    @patch('benchmarks.itbench.client.subprocess.run')
    def test_malformed_actions_survive_durable_journal_recovery(self, run):
        directory = Path(self.temp.name)
        self.client.journal_path = directory / 'events.jsonl'
        actions = [['exec_kubectl', ['get', 'pods']], {'command': 'get pods'}]
        for action in actions:
            self.client.execute_action(action)
        events = [json.loads(line) for line in self.client.journal_path.read_text().splitlines()]
        self.assertEqual([e['kind'] for e in events],
                         ['tool_action_started', 'tool_call'] * 2)
        append_event(self.client.journal_path, 'tool_action_started', {
            'timestamp': 'in-flight', 'role': 'probe', 'action': ['get_alerts']})
        record = recover_trial(directory, 53, {}, 'cancelled')
        self.assertEqual([c['action'] for c in record['tool_calls']], actions)
        self.assertTrue(all(c['result']['error'] for c in record['tool_calls']))
        self.assertEqual([c['action'] for c in record['in_flight_tool_calls']], [['get_alerts']])
        self.assertFalse(record['record_complete'])
        self.assertIsNone(record['success'])
        run.assert_not_called()
