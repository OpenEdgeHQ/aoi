import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from benchmarks.itbench.batch import Batch, StageError, catalog, write_json


class BatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.itbench = self.root / 'ITBench'
        self.create_scenario(16, 'sre', 'environment-fault')
        self.create_scenario(70, 'finops', 'cost-fault')

    def tearDown(self):
        self.temp.cleanup()

    def create_scenario(self, number, domain, fault):
        library = self.itbench / f'library/scenarios/{number}/scenario.yaml'
        archive = self.itbench / f'archive/scenarios/{number}/scenario.yaml'
        library.parent.mkdir(parents=True, exist_ok=True)
        archive.parent.mkdir(parents=True, exist_ok=True)
        library.write_text(yaml.safe_dump({'environment': {'domains': [domain],
            'applications': [{'id': 'opentelemetry-demo'}]}, 'faults': [{'injections': [{'id': fault}]}]}))
        archive.write_text(yaml.safe_dump({'faults': [{'injections': [{'id': fault}]}]}))

    def test_catalog_only_selects_library_sre(self):
        cases = catalog(self.itbench)
        self.assertEqual([case['scenario_number'] for case in cases], [16])
        self.assertEqual(cases[0]['namespaces'], ['otel-demo'])
        self.assertNotIn('environment-fault', cases[0]['incident'])
        self.assertEqual(cases[0]['deployment_variables']['applications_opentelemetry_demo_enabled'], True)

    def test_executable_fault_mismatch_is_rejected(self):
        (self.itbench / 'archive/scenarios/16/scenario.yaml').write_text('faults: []')
        with self.assertRaises(ValueError):
            catalog(self.itbench)

    def test_library_deployment_options_are_preserved(self):
        (self.itbench / 'library/scenarios/16/extra_variables.yaml').write_text('applications_opentelemetry_demo_load_generator_users: 7')
        self.assertEqual(catalog(self.itbench)[0]['deployment_variables']['applications_opentelemetry_demo_load_generator_users'], 7)

    def test_atomic_progress_write(self):
        destination = self.root / 'results/progress.json'
        write_json(destination, {'state': 'running'})
        write_json(destination, {'state': 'completed'})
        self.assertEqual(json.loads(destination.read_text())['state'], 'completed')
        self.assertFalse(destination.with_suffix('.json.tmp').exists())

    @patch('benchmarks.itbench.batch.subprocess.check_output')
    def test_preflight_refuses_another_cluster(self, command):
        admin = self.root / 'kubeconfig'
        admin.write_text('current-context: kind-kind')
        args = argparse.Namespace(itbench_dir=self.itbench, admin_kubeconfig=admin,
            output=self.root / 'output', itbench_python='python3.13', docker_config='config')
        batch = Batch(args)
        with self.assertRaises(ValueError):
            batch.check_cluster()
        command.assert_not_called()

    def test_inventory_cannot_redirect_cleanup_to_another_cluster(self):
        admin = self.root / 'admin'
        inventory = self.itbench / 'scenarios/sre/inventory/group_vars/environment/cluster.yaml'
        inventory.parent.mkdir(parents=True)
        inventory.write_text('cluster:\n  kubeconfig: /other/kubeconfig')
        batch = Batch(argparse.Namespace(itbench_dir=self.itbench, admin_kubeconfig=admin,
            output=self.root / 'output', itbench_python='python3.13', docker_config='config'))
        with self.assertRaises(StageError):
            batch.make('remove-fault-resources', 16, self.root / 'cleanup.log')

    @patch('benchmarks.itbench.batch.subprocess.check_output', return_value='')
    def test_missing_target_is_an_environment_error_before_model_trial(self, command):
        source = self.itbench / 'library/scenarios/16/scenario.yaml'
        source.write_text(yaml.safe_dump({'faults': [{'injections': [{'targets': [{'kubernetes': {
            'kind': 'Deployment', 'name': 'missing', 'namespace': 'otel-demo'}}]}]}]}))
        batch = Batch.__new__(Batch)
        batch.admin = self.root / 'admin'
        with self.assertRaisesRegex(StageError, 'model trial not started'):
            batch.check_targets({'catalog_source': str(source)})
        self.assertIn(str(batch.admin), command.call_args.args[0])
