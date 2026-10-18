import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.itbench.batch import Batch, build_parser, experiment_provenance, validate_resume, main
from benchmarks.itbench.pricing import estimate_cost, validate_prices
from benchmarks.itbench.summary import summarize


class PricingTest(unittest.TestCase):
    def command(self, model, input_rate=None, output_rate=None):
        args = build_parser().parse_args([
            '--itbench-dir', '/unused/itbench', '--admin-kubeconfig', '/unused/config',
            '--output', '/unused/results', '--model', model,
            *([] if input_rate is None else ['--input-usd-per-million', str(input_rate)]),
            *([] if output_rate is None else ['--output-usd-per-million', str(output_rate)]),
        ])
        batch = Batch.__new__(Batch)
        batch.args = args
        batch.itbench = args.itbench_dir
        return batch.model_command({'scenario_number': 16, 'incident': 'Investigate'},
                                   Path('/unused/access'), args.output)

    def test_qwen_and_default_model_have_no_implicit_prices(self):
        usage = {'source': 'provider_reported', 'total': {'input_tokens': 1000000, 'output_tokens': 1000000}}
        for model in ('Qwen/Qwen3-14B', 'gpt-4o-mini'):
            with self.subTest(model=model):
                command = self.command(model)
                self.assertEqual(command[command.index('--model') + 1], model)
                self.assertNotIn('--input-usd-per-million', command)
                self.assertNotIn('--output-usd-per-million', command)
                self.assertIsNone(estimate_cost(usage, None, None))

    def test_explicit_prices_reach_task_runner(self):
        command = self.command('Qwen/Qwen3-14B', 0.2, 0.8)
        self.assertEqual(command[command.index('--input-usd-per-million') + 1], '0.2')
        self.assertEqual(command[command.index('--output-usd-per-million') + 1], '0.8')
        usage = {'source': 'provider_reported', 'total': {'input_tokens': 1000000, 'output_tokens': 500000}}
        self.assertEqual(estimate_cost(usage, 0.2, 0.8), 0.6)
        self.assertEqual(estimate_cost(usage, 0, 0), 0)
        self.assertIsNone(estimate_cost({**usage, 'source': 'unavailable'}, 0.2, 0.8))

    def test_invalid_or_partial_prices_are_rejected(self):
        for rates in ((None, 1), (1, None), (-1, 1), (float('nan'), 1), (1, float('inf'))):
            with self.subTest(rates=rates), self.assertRaises(ValueError):
                validate_prices(*rates)


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.aoi, self.itbench = self.root / 'aoi', self.root / 'itbench'
        for repo in (self.aoi, self.itbench):
            repo.mkdir()
            self.git(repo, 'init', '-q')
            (repo / 'tracked.txt').write_text('baseline\n')
            self.git(repo, 'add', '.')
            self.git(repo, 'commit', '-qm', 'baseline')
        self.configuration, self.cases = {'model': 'Qwen/Qwen3-14B'}, []
        self.previous = {'configuration': self.configuration, 'cases': self.cases,
                         **experiment_provenance(self.aoi, self.itbench)}

    def git(self, repo, *args):
        return subprocess.check_output(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                                        '-c', 'commit.gpgsign=false', '-C', str(repo), *args])

    def validate(self, previous=None):
        validate_resume(previous if previous is not None else self.previous, self.configuration,
                        self.cases, experiment_provenance(self.aoi, self.itbench))

    def test_unchanged_repositories_can_resume(self):
        self.validate()

    def test_new_commit_in_either_repository_blocks_resume(self):
        for name, repo in (('aoi', self.aoi), ('itbench', self.itbench)):
            with self.subTest(repo=name):
                self.git(repo, 'commit', '--allow-empty', '-qm', 'new version')
                with self.assertRaisesRegex(ValueError, name + '_revision'):
                    self.validate()
                self.previous.update(experiment_provenance(self.aoi, self.itbench))

    def test_staged_and_unstaged_patch_changes_block_resume(self):
        for repo in (self.aoi, self.itbench):
            for staged in (False, True):
                with self.subTest(repo=repo.name, staged=staged):
                    (repo / 'tracked.txt').write_text(f'changed {staged}\n')
                    if staged:
                        self.git(repo, 'add', '.')
                    with self.assertRaisesRegex(ValueError, repo.name + '_patch_sha256'):
                        self.validate()
                    self.git(repo, 'reset', '--hard', '-q', 'HEAD')

    def test_legacy_manifest_requires_new_output_directory(self):
        for key in ('aoi_revision', 'itbench_revision', 'itbench_patch_sha256', 'aoi_patch_sha256'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'new output directory'):
                self.validate({k: v for k, v in self.previous.items() if k != key})

    def test_resume_configuration_and_catalog_still_checked(self):
        for key in ('configuration', 'cases'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate({**self.previous, key: None})

    def test_rejected_cli_resume_does_not_rewrite_previous_reports(self):
        output = self.root / 'saved-results'
        output.mkdir()
        files = [output / name for name in ('progress.json', 'summary.json', 'report.md')]
        for path in files:
            path.write_text('previous run evidence')
        arguments = ['batch', '--itbench-dir', str(self.itbench), '--admin-kubeconfig', 'unused',
                     '--output', str(output), '--itbench-python', 'unused', '--docker-config', 'unused']
        with patch('sys.argv', arguments), patch('benchmarks.itbench.batch.Batch') as constructor:
            batch = constructor.return_value
            batch.execute.side_effect = ValueError('Resume mismatch; use a new output directory')
            batch.manifest_validated = False
            batch.forward = None
            with self.assertRaises(ValueError):
                main()
            batch.report.assert_not_called()
        for path in files:
            self.assertEqual(path.read_text(), 'previous run evidence')

    def test_manifest_records_backend_and_resume_blocks_changed_endpoint_before_deployment(self):
        batch = Batch.__new__(Batch)
        batch.root, batch.itbench = self.aoi, self.itbench
        batch.output = self.root / 'backend-output'
        batch.cases = [{'scenario_number': 16}]
        batch.env = {}
        batch.args = argparse.Namespace(model='Qwen/Qwen3-14B', input_usd_per_million=None,
                                        output_usd_per_million=None, settle_seconds=30, model_timeout=1800)
        first = {'api_source': 'openai', 'llm_provider': 'openai', 'api_base': 'http://localhost:8000/v1'}
        with patch.object(batch, 'check_cluster'), patch.object(batch, 'phase'), \
             patch('benchmarks.itbench.batch.model_backend_identity', return_value=first), \
             patch.object(batch, 'make', side_effect=RuntimeError('stop before deployment')):
            with self.assertRaisesRegex(RuntimeError, 'stop before deployment'):
                batch.execute()
        manifest = batch.output / 'manifest.json'
        original = manifest.read_text()
        saved = json.loads(original)['configuration']
        for key, value in first.items():
            self.assertEqual(saved[key], value)
        self.assertEqual(batch.env['API_BASE'], first['api_base'])
        self.assertEqual(batch.env['API_SOURCE'], first['api_source'])
        second = {**first, 'api_base': 'http://localhost:9000/v1'}
        with patch.object(batch, 'check_cluster'), \
             patch('benchmarks.itbench.batch.model_backend_identity', return_value=second), \
             patch.object(batch, 'make') as deploy:
            with self.assertRaisesRegex(ValueError, 'configuration.*new output directory'):
                batch.execute()
            deploy.assert_not_called()
        self.assertEqual(manifest.read_text(), original)

    def test_mismatch_stops_before_deployment_and_preserves_manifest(self):
        batch = Batch.__new__(Batch)
        batch.root, batch.itbench = self.aoi, self.itbench
        batch.output = self.root / 'output'
        batch.output.mkdir()
        batch.cases = self.cases
        batch.args = argparse.Namespace(model='Qwen/Qwen3-14B', input_usd_per_million=None,
                                        output_usd_per_million=None, settle_seconds=30, model_timeout=1800)
        path = batch.output / 'manifest.json'
        original = json.dumps(self.previous)
        path.write_text(original)
        with patch.object(batch, 'check_cluster'), patch.object(batch, 'make') as deploy:
            with self.assertRaisesRegex(ValueError, 'new output directory'):
                batch.execute()
            deploy.assert_not_called()
        self.assertEqual(path.read_text(), original)
        self.assertFalse((batch.output / 'pid.json').exists())


class ExecutionSummaryTest(unittest.TestCase):
    def record(self, success, duration, calls, tokens, cost):
        return {'success': success, 'duration_seconds': duration, 'tool_calls': [{}] * calls,
                'token_usage': {'source': 'provider_reported', 'total': {'total_tokens': tokens}},
                'cost_usd': cost}

    def test_ungraded_runs_have_statistics_but_no_success_rate(self):
        result = summarize([self.record(None, 10, 2, 100, None), self.record(None, 30, 4, 300, None)])
        stats = result['execution_stats']
        self.assertEqual((stats['attempts'], stats['mean_duration_seconds'], stats['mean_tool_calls'],
                          stats['mean_total_tokens']), (2, 20, 3, 200))
        self.assertIsNone(stats['mean_cost_usd'])
        self.assertEqual(stats['cost_coverage'], 0)
        self.assertIsNone(result['success_rate'])
        self.assertEqual(result['verified_attempts'], 0)
        self.assertIsNone(result['mean_duration_seconds'])

    def test_mixed_results_preserve_graded_denominator_and_partial_cost_coverage(self):
        records = [self.record(True, 10, 2, 100, 0.2), self.record(False, 20, 4, 200, None),
                   self.record(None, 60, 9, 600, 0.4)]
        result = summarize(records)
        self.assertEqual(result['success_rate'], 0.5)
        self.assertEqual(result['mean_duration_seconds'], 15)
        self.assertEqual(result['cost_coverage'], 1)
        stats = result['execution_stats']
        self.assertEqual(stats['mean_duration_seconds'], 30)
        self.assertEqual(stats['mean_tool_calls'], 5)
        self.assertEqual(stats['mean_cost_usd'], 0.3)
        self.assertEqual(stats['cost_coverage'], 2)

    def test_missing_usage_is_not_zero_and_failed_attempts_are_included(self):
        records = [self.record(None, 10, 0, 0, 0),
                   {'success': None, 'error': 'provider unavailable', 'duration_seconds': 20,
                    'token_usage': {'source': 'unavailable', 'total': {'total_tokens': 0}}}]
        result = summarize(records)['execution_stats']
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['mean_duration_seconds'], 15)
        self.assertEqual(result['token_coverage'], 1)
        self.assertEqual(result['tool_calls_coverage'], 1)
        self.assertEqual(result['mean_cost_usd'], 0)

    def test_empty_summary_has_null_means(self):
        result = summarize([])
        self.assertEqual(result['execution_stats']['attempts'], 0)
        for key, value in result['execution_stats'].items():
            if key.startswith('mean_'):
                self.assertIsNone(value)
        self.assertIsNone(result['success_rate'])
