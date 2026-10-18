import argparse
import asyncio
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from benchmarks.itbench.preflight import check_permissions, require_permissions
from benchmarks.itbench.batch import Batch


class PermissionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'archive/scenarios/48/scenario.yaml'
        self.source.parent.mkdir(parents=True)
        self.source.write_text(yaml.safe_dump({'faults': [{'injections': [{
            'id': 'disabled-istio-ambient-mode-kubernetes-namespace',
            'args': {'kubernetesObject': {'kind': 'Namespace', 'metadata': {'name': 'book-info'}}}
        }]}]}))
        self.access = self.root / 'restricted'
        self.access.write_text('test')

    @patch('benchmarks.itbench.preflight.subprocess.run')
    def test_namespace_patch_denied_blocks_and_writes_evidence(self, run):
        run.side_effect = [subprocess.CompletedProcess([], 0, 'yes\n', ''),
                           subprocess.CompletedProcess([], 0, 'yes\n', ''),
                           subprocess.CompletedProcess([], 1, 'no\n', '')]
        output = self.root / 'permission.json'
        with self.assertRaisesRegex(ValueError, 'model trial not started.*patch namespaces/book-info'):
            require_permissions(self.root, 48, self.access, ['book-info'], output)
        report = json.loads(output.read_text())
        self.assertFalse(report['passed'])
        self.assertFalse(report['model_executed'])
        self.assertEqual(run.call_args.args[0], ['kubectl', '--kubeconfig', str(self.access),
                                                'auth', 'can-i', 'patch', 'namespaces/book-info'])

    @patch('benchmarks.itbench.preflight.subprocess.run')
    def test_all_allowed_and_unknown_fault_coverage_is_explicit(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, 'yes\n', '')
        result = check_permissions(self.root, 48, self.access, ['book-info'])
        self.assertTrue(result['passed'])
        self.assertEqual(result['coverage'], 'known_repair_operations')
        self.source.write_text('faults: [{injections: [{id: future-fault}]}]')
        result = check_permissions(self.root, 48, self.access, ['book-info'])
        self.assertEqual(result['coverage'], 'partial')
        self.assertEqual(result['unchecked_fault_ids'], ['future-fault'])

    @patch('benchmarks.itbench.preflight.subprocess.run')
    def test_auth_failure_or_timeout_is_not_permission_success(self, run):
        for failure in (subprocess.CompletedProcess([], 1, '', 'Unauthorized'),
                        subprocess.TimeoutExpired('kubectl', 30)):
            with self.subTest(failure=failure):
                run.side_effect = failure if isinstance(failure, Exception) else None
                run.return_value = failure
                result = check_permissions(self.root, 48, self.access, ['book-info'])
                self.assertFalse(result['passed'])
                self.assertTrue(all(c['detail'] for c in result['checks']))

    @patch('benchmarks.itbench.preflight.subprocess.run')
    def test_pvc_repair_checks_create_delete_and_deployment_patch(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, 'yes\n', '')
        self.source.write_text(yaml.safe_dump({'faults': [{'injections': [{
            'id': 'nonexistent-kubernetes-workload-persistent-volume-claim',
            'args': {'kubernetesObject': {'kind': 'Deployment', 'metadata': {
                'name': 'reviews-v3', 'namespace': 'book-info'}}}
        }]}]}))
        result = check_permissions(self.root, 48, self.access, ['book-info'])
        operations = {(c['verb'], c['resource'], c.get('namespace')) for c in result['checks']}
        self.assertIn(('patch', 'deployments.apps/reviews-v3', 'book-info'), operations)
        for verb in ('get', 'create', 'delete'):
            self.assertIn((verb, 'persistentvolumeclaims', 'book-info'), operations)

    def test_runner_blocks_before_model_configuration(self):
        from benchmarks.itbench.run import run
        args = argparse.Namespace(input_usd_per_million=None, output_usd_per_million=None,
            scenario_number=48, itbench_dir=self.root, kubeconfig=self.access,
            start_scenario=False, incident='Investigate', prometheus_url=None,
            namespaces=['book-info'], output=self.root / 'task.json')
        with patch('benchmarks.itbench.run.require_permissions', side_effect=ValueError('blocked')), \
             patch('benchmarks.itbench.run._configure_model') as configure:
            with self.assertRaisesRegex(ValueError, 'blocked'):
                asyncio.run(run(args))
            configure.assert_not_called()
        self.assertFalse(args.output.exists())

    def test_batch_permission_failure_never_injects_or_starts_model(self):
        self.check_batch_permission_failure()

    def test_alert_evaluation_permission_failure_never_injects_or_starts_model(self):
        self.check_batch_permission_failure(mitigation=True)

    def check_batch_permission_failure(self, mitigation=False):
        batch = Batch.__new__(Batch)
        batch.root = self.root
        batch.itbench = self.root
        batch.output = self.root / 'output'
        (batch.output / 'scenarios/48').mkdir(parents=True)
        batch.admin = self.root / 'admin'
        batch.env = {}
        batch.cases = [{'scenario_number': 48, 'namespaces': ['book-info'], 'deployment_variables': {}}]
        batch.progress = {'completed': []}
        ground_truth = self.root / 'library/scenarios/48/ground_truth.yaml'
        ground_truth.parent.mkdir(parents=True)
        ground_truth.write_text('alerts: [{name: Fault}]')
        batch.args = argparse.Namespace(model='test', input_usd_per_million=None,
            output_usd_per_million=None, settle_seconds=0, model_timeout=10, uv_command='uv', mitigation_evaluation=mitigation)
        access = self.root / 'res/itbench/runtime/kubeconfig'
        access.parent.mkdir(parents=True); access.write_text('test')
        with patch.object(batch, 'check_cluster'), patch.object(batch, 'phase'), \
             patch.object(batch, 'make') as make, patch.object(batch, 'start_forward'), \
             patch.object(batch, 'cleanup'), patch.object(batch, 'command') as command, \
             patch.object(batch, 'check_targets'), patch.object(batch, 'snapshot'), \
             patch.object(batch, 'report'), patch.object(batch, 'model_command') as model, \
             patch.object(batch, 'collect_alerts', return_value={'telemetry_healthy': True, 'alerts': []}) as alerts, \
             patch('benchmarks.itbench.batch.model_backend_identity', return_value={'api_source':'openai','api_base':'test'}), \
             patch('benchmarks.itbench.batch.experiment_provenance', return_value={}), \
             patch('benchmarks.itbench.batch.require_permissions', side_effect=ValueError('permission denied')):
            batch.execute()
            model.assert_not_called()
            self.assertEqual(alerts.call_count, int(mitigation))
            self.assertEqual(command.call_count, 2)  # deploy + baseline readiness only
            self.assertNotIn('inject-scenario-faults', [c.args[0] for c in make.call_args_list])
        ledger = json.loads((batch.output / 'scenarios/48/case.json').read_text())
        self.assertEqual(ledger['status'], 'environment_error')
        self.assertIn('permission denied', ledger['error'])
