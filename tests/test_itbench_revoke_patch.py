"""Apply the shipped upstream patch to a pinned revoke fixture before checking it."""
import subprocess
import tempfile
import unittest
from pathlib import Path
import yaml


class RevokePatchTest(unittest.TestCase):
    def test_patch_cleans_all_namespaces_with_both_ownership_labels_before_account_deletion(self):
        root = Path(__file__).resolve().parents[1]
        relative = 'scenarios/sre/project/roles/agent/tasks/revoke.yaml'
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            target = directory / relative; target.parent.mkdir(parents=True)
            target.write_bytes((root / 'tests/fixtures/itbench-revoke-333afa05.yaml').read_bytes())
            subprocess.run(['git', 'apply', '--include=' + relative,
                            str(root / 'benchmarks/itbench/patches/itbench-333afa05.patch')],
                           cwd=directory, check=True, capture_output=True)
            tasks = yaml.safe_load(target.read_text())
        query = tasks[0]['kubernetes.core.k8s_info']
        self.assertEqual(tasks[0]['loop'], ['RoleBinding', 'Role'])
        self.assertNotIn('namespace', query)  # include old scenario namespaces
        self.assertCountEqual(query['label_selectors'], [
            'app.kubernetes.io/component = agent-access-provider',
            'app.kubernetes.io/managed-by = ITBench'])
        deletion = tasks[1]['kubernetes.core.k8s']
        self.assertEqual(deletion['state'], 'absent')
        for field in ('kind', 'name', 'namespace'):
            self.assertIn('item.', deletion[field])
        self.assertIn("map(attribute='resources')", tasks[1]['loop'])
        self.assertEqual(tasks[-1]['kubernetes.core.k8s']['kind'], 'ServiceAccount')
        self.assertTrue(all(task.get('kubernetes.core.k8s', {}).get('state', 'absent') == 'absent'
                            for task in tasks))
