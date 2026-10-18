import copy
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.itbench.evaluate import verdict, namespace_matches
from benchmarks.itbench.journal import append_event
from benchmarks.itbench.recover import recover_trial
from benchmarks.itbench.summary import summarize


class AlertEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.clean = {"telemetry_healthy": True, "rule_healthy": True, "alerts": [], "rules": [{"name": "Fault", "query": "errors > 0"}]}
        self.fault = {**self.clean, "alerts": [{"state": "pending"}]}
        self.state = {"app": {"items": [{"kind": "Deployment", "metadata": {"name": "worker", "generation": 2},
            "spec": {"replicas": 1}, "status": {"readyReplicas": 1, "updatedReplicas": 1, "observedGeneration": 2}}]}}

    def check(self, before=None, injected=None, post=None, state=None):
        return verdict(before or self.clean, injected or self.fault, post or [self.clean]*3, self.state, state or self.state)

    def test_observed_fault_then_sustained_resolution(self):
        r = self.check()
        self.assertTrue(r["passed"])
        self.assertFalse(r["official"])
        self.assertIsNone(r["official_success"])

    def test_never_observed_fault_cannot_pass(self):
        self.assertIsNone(self.check(injected=self.clean)["passed"])

    def test_preexisting_fault_is_not_a_valid_baseline(self):
        self.assertIsNone(self.check(before=self.fault)["passed"])

    def test_persistent_fault_fails(self):
        self.assertFalse(self.check(post=[self.fault]*3)["passed"])

    def test_missing_telemetry_and_deleted_rules_cannot_pass(self):
        self.assertIsNone(self.check(post=[{**self.clean, "telemetry_healthy": False}]*3)["passed"])
        self.assertIsNone(self.check(post=[{**self.clean, "rules": []}]*3)["passed"])

    def test_disabling_application_and_stale_ready_replicas_cannot_pass(self):
        disabled = copy.deepcopy(self.state);disabled["app"]["items"][0]["spec"]["replicas"] = 0
        self.assertFalse(self.check(state=disabled)["passed"])
        stale = copy.deepcopy(self.state);stale["app"]["items"][0]["status"]["observedGeneration"] = 1
        self.assertFalse(self.check(state=stale)["passed"])

    def test_unrelated_namespace_is_not_a_fault_alert(self):
        self.assertFalse(namespace_matches({"namespace": "kube-system"}, ["app"]))
        self.assertTrue(namespace_matches({"destination_workload_namespace": "app"}, ["app"]))


class RecoveryTest(unittest.TestCase):
    def test_partial_trial_preserves_usage_outputs_and_inflight_call(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d);j = p/"events.jsonl"
            append_event(j, "trial_started", {"started_at": "2026-10-11T00:00:00+00:00"})
            append_event(j, "provider_usage", {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "role": "probe"})
            append_event(j, "tool_action_started", {"timestamp": "t", "role": "probe", "action": "get_alerts()"})
            with j.open('a') as f: f.write('{unfinished')
            r = recover_trial(p, 116, {"input_usd_per_million": .15, "output_usd_per_million": .6}, "killed")
            self.assertEqual(r["token_usage"]["total"]["total_tokens"], 110)
            self.assertEqual(r["cost_usd"], .000021)
            self.assertEqual(len(r["in_flight_tool_calls"]), 1)
            self.assertFalse(r["record_complete"])
            self.assertIsNone(r["success"])
            self.assertEqual(summarize([r])["partial_record_count"], 1)

    def test_absent_journal_usage_remains_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            r = recover_trial(Path(d), 116, {}, "killed")
            self.assertIsNone(r["token_usage"]["total"])
            self.assertIsNone(r["cost_usd"])
            self.assertEqual(summarize([r])["execution_stats"]["token_coverage"], 0)
