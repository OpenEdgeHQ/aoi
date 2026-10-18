"""Exercise actual llm_config loading in fresh processes, as batch restarts do."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks.itbench.batch import validate_resume


class BackendIdentityTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'llm_config.py', self.root / 'llm_config.py')

    def identity(self, overrides=None):
        env = {key: value for key, value in os.environ.items()
               if key not in {'API_SOURCE', 'API_BASE', 'API_KEY', 'OPENAI_ENDPOINT',
                              'PYTHONPATH', 'PYTHON_DOTENV_DISABLED'}}
        env.update(overrides or {})
        output = subprocess.check_output(
            [sys.executable, '-c', 'import json, llm_config; print(json.dumps(llm_config.backend_identity()))'],
            cwd=self.root, env=env, text=True)
        return json.loads(output)

    def check_resume(self, first, second):
        provenance = {'aoi_revision': 'same', 'itbench_revision': 'same', 'itbench_patch_sha256': 'same'}
        configuration = {'model': 'Qwen/Qwen3-14B', **first}
        validate_resume({'configuration': configuration, 'cases': [], **provenance},
                        {'model': 'Qwen/Qwen3-14B', **second}, [], provenance)

    def test_environment_endpoint_change_rejects_resume_with_same_model(self):
        first = self.identity({'API_BASE': 'http://localhost:8000/v1'})
        second = self.identity({'API_BASE': 'http://localhost:9000/v1'})
        self.assertEqual(first['api_base'], 'http://localhost:8000/v1')
        with self.assertRaisesRegex(ValueError, 'configuration.*new output directory'):
            self.check_resume(first, second)

    def test_provider_change_rejects_resume_with_same_endpoint(self):
        common = {'API_BASE': 'https://example.invalid/v1'}
        first = self.identity({**common, 'API_SOURCE': 'openai'})
        second = self.identity({**common, 'API_SOURCE': 'openrouter'})
        self.assertEqual(first['llm_provider'], 'openai')
        with self.assertRaises(ValueError):
            self.check_resume(first, second)

    def test_dotenv_change_rejects_resume_and_environment_takes_precedence(self):
        dotenv = self.root / '.env'
        dotenv.write_text('API_SOURCE=openai\nAPI_BASE=http://localhost:8000/v1\nAPI_KEY=secret-from-file\n')
        first = self.identity()
        self.assertEqual(first['api_source'], 'openai')
        self.assertEqual(first['api_base'], 'http://localhost:8000/v1')
        dotenv.write_text('API_SOURCE=openai\nAPI_BASE=http://localhost:9000/v1\nAPI_KEY=secret-from-file\n')
        with self.assertRaises(ValueError):
            self.check_resume(first, self.identity())
        overridden = self.identity({'API_BASE': 'http://localhost:8000/v1'})
        self.check_resume(first, overridden)

    def test_defaults_and_provider_endpoint_fallback(self):
        defaults = self.identity()
        self.assertEqual(defaults, {'api_source': 'openrouter', 'llm_provider': 'openai',
                                    'api_base': 'https://openrouter.ai/api/v1'})
        empty = self.identity({'API_BASE': ''})
        self.assertEqual(empty['api_base'], 'https://api.openai.com/v1')
        custom = self.identity({'API_BASE': '', 'OPENAI_ENDPOINT': 'https://example.invalid/v1'})
        self.assertEqual(custom['api_base'], 'https://example.invalid/v1')
        with self.assertRaises(ValueError):
            self.check_resume(empty, custom)

    def test_unchanged_backend_and_key_rotation_allow_resume_without_leaking_keys(self):
        first = self.identity({'API_KEY': 'first-secret'})
        second = self.identity({'API_KEY': 'second-secret'})
        self.check_resume(first, second)
        self.assertEqual(set(first), {'api_source', 'llm_provider', 'api_base'})
        self.assertNotIn('secret', json.dumps([first, second]))
