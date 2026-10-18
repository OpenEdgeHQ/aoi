import copy
import json
import unittest

from benchmarks.itbench.verify import check_scenario_one
from utils.json_response import parse_model_json


class VerificationTest(unittest.TestCase):
    def setUp(self):
        self.flag = {'defaultVariant': 'on', 'state': 'ENABLED', 'variants': {'on': 100, 'off': 0}}
        self.deployment = {'metadata': {'generation': 2}, 'spec': {'replicas': 1},
                           'status': {'readyReplicas': 1, 'updatedReplicas': 1, 'observedGeneration': 2}}

    def check(self, flag=None, deployments=None):
        configmap = {'data': {'demo.flagd.json': json.dumps({'flags': {'loadGeneratorFloodHomepage': flag or self.flag}})}}
        return check_scenario_one(configmap, deployments or {n: self.deployment for n in ('flagd', 'load-generator')})

    def test_active_fault_fails(self):
        self.assertFalse(self.check()['passed'])
        self.assertFalse(self.check()['official'])

    def test_disabled_flag_and_healthy_workloads_pass(self):
        flag = {**self.flag, 'defaultVariant': 'off'}
        self.assertTrue(self.check(flag)['passed'])

    def test_scaling_load_generator_to_zero_is_not_a_repair(self):
        stopped = copy.deepcopy(self.deployment)
        stopped['spec']['replicas'] = 0
        stopped['status']['readyReplicas'] = 0
        self.assertFalse(self.check({**self.flag, 'state': 'DISABLED'}, {
            'flagd': self.deployment, 'load-generator': stopped})['passed'])

    def test_old_rollout_is_not_ready(self):
        old = copy.deepcopy(self.deployment)
        old['status']['observedGeneration'] = 1
        self.assertFalse(self.check({**self.flag, 'state': 'DISABLED'}, {
            'flagd': old, 'load-generator': self.deployment})['passed'])

    def test_invalid_apostrophe_escape_becomes_valid_command(self):
        self.assertEqual(parse_model_json(r'''{"command": "echo \'quoted\'"}''')['command'], "echo 'quoted'")

    def test_invalid_regex_escape_keeps_literal_backslash(self):
        self.assertEqual(parse_model_json(r'''{"pattern": "\d+"}''')['pattern'], r'\d+')

    def test_json_syntax_errors_are_not_silently_accepted(self):
        with self.assertRaises(json.JSONDecodeError):
            parse_model_json('{"command": "unterminated}')


if __name__ == '__main__':
    unittest.main()
