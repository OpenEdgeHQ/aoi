import argparse
import ast
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from benchmarks.itbench.batch import Batch
from benchmarks.itbench.client import ITBenchEnvironmentClient


class ToolContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.kubeconfig = self.root / 'restricted-config'
        self.kubeconfig.write_text('test')
        self.client = self.client_for(['book-info'])

    def client_for(self, namespaces):
        return ITBenchEnvironmentClient('scenario', 'Investigate current symptoms',
                                        self.kubeconfig, namespaces=namespaces)

    def test_examples_use_scenario_metadata_and_remain_parseable(self):
        for namespaces in (['book-info'], ['otel-demo'], ['book-info', 'otel-demo']):
            with self.subTest(namespaces=namespaces):
                task = self.client_for(namespaces).init_problem('scenario')
                commands = re.findall(r'exec_kubectl\(\[.*?\]\)', task['instructions'])
                self.assertEqual(len(commands), len(namespaces))
                for command, namespace in zip(commands, namespaces):
                    args = ast.literal_eval(ast.parse(command, mode='eval').body.args[0])
                    self.assertEqual(args, ['get', 'pods', '-n', namespace])
                self.assertEqual(task['namespaces'], namespaces)
                if namespaces == ['book-info']:
                    self.assertNotIn('otel-demo', task['instructions'])

    def test_no_metadata_does_not_invent_a_namespace(self):
        task = self.client_for(None).init_problem('scenario')
        self.assertEqual(task['namespaces'], [])
        self.assertNotIn('otel-demo', task['instructions'])
        self.assertIn('replace <incident-namespace>', task['instructions'])

    def test_batch_passes_each_catalog_namespace(self):
        batch = Batch.__new__(Batch)
        batch.args = argparse.Namespace(model='gpt-4o-mini', prometheus_port=19090,
                                        input_usd_per_million=None, output_usd_per_million=None)
        batch.itbench = self.root
        command = batch.model_command({'scenario_number': 48, 'incident': 'Investigate',
                                       'namespaces': ['book-info', 'otel-demo']}, self.kubeconfig, self.root)
        self.assertEqual([command[i+1] for i, arg in enumerate(command) if arg == '--namespace'],
                         ['book-info', 'otel-demo'])

    @patch('benchmarks.itbench.client.subprocess.run')
    def test_apply_accepts_stdin_before_or_after_namespace_and_long_option(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = 'configured'
        run.return_value.stderr = ''
        manifest = 'apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: example\n'
        for args in (
            ['apply', '-f', '-', '-n', 'book-info'],
            ['apply', '-n', 'book-info', '-f', '-'],
            ['-n', 'book-info', 'apply', '-f', '-', '--validate=false'],
            ['apply', '--filename', '-', '--namespace=book-info'],
            ['apply', '--filename=-', '-n', 'book-info'],
            ['apply', '-f=-', '-n', 'book-info'],
            ['apply', '-f-', '-n', 'book-info'],
        ):
            with self.subTest(args=args):
                result = self.client.execute_action(f'exec_kubectl({args!r}, stdin={manifest!r})')
                self.assertNotIn('error', result)
                self.assertEqual(run.call_args.args[0], ['kubectl', '--kubeconfig', str(self.kubeconfig.resolve()), *args])
                self.assertEqual(run.call_args.kwargs['input'], manifest)
                self.assertNotIn('shell', run.call_args.kwargs)

    @patch('benchmarks.itbench.client.subprocess.run')
    def test_invalid_stdin_forms_and_privilege_overrides_remain_rejected(self, run):
        for args in (
            ['apply', '-f'], ['apply'], ['apply', '-f', 'local.yaml'],
            ['apply', '-f', '-', '-f', 'other.yaml'], ['apply', '-f', '-', '--filename=-'],
            ['apply', '-k', 'directory', '-f', '-'],
            ['delete', '-f', '-'], ['get', 'pods', '-f', '-'],
            ['apply', '-f', '-', '--kubeconfig', '/admin'],
            ['apply', '-f', '-', '--context=admin'],
        ):
            with self.subTest(args=args):
                result = self.client.execute_action(f'exec_kubectl({args!r}, stdin="manifest")')
                self.assertTrue(result.get('error'))
        run.assert_not_called()

    @patch('benchmarks.itbench.client.subprocess.run')
    def test_probe_cannot_apply_regardless_of_argument_order(self, run):
        result = self.client.execute_probe_action(
            "exec_kubectl(['apply', '-f', '-', '-n', 'book-info'], stdin='manifest')")
        self.assertIn('Probe cannot', result)
        run.assert_not_called()

    def test_initial_probe_check_has_no_obsolete_show_all_flag(self):
        prompts = yaml.safe_load((Path(__file__).resolve().parents[1] / 'prompts/probe_prompts.yaml').read_text())
        system = prompts['system_prompt']
        self.assertIn('kubectl get pods -n <namespace>', system)
        self.assertNotIn('--show-all', system)
