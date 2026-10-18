import unittest

from benchmarks.itbench.review import review


class SnapshotReviewTest(unittest.TestCase):
    def snapshot(self, replicas):
        return {"app": {"items": [{"kind": "Deployment", "metadata": {"name": "worker", "namespace": "app", "generation": 2},
                                  "spec": {"replicas": replicas}, "status": {"readyReplicas": replicas, "observedGeneration": 2}}]}}

    def test_running_other_workloads_does_not_hide_scaled_to_zero_fault(self):
        result = review(self.snapshot(1), self.snapshot(0), self.snapshot(0))
        self.assertFalse(result["all_changed_fields_match_baseline"])
        self.assertIsNone(result["official_success"])
        self.assertFalse(result["official"])

    def test_restored_configuration_is_still_not_official_success(self):
        result = review(self.snapshot(1), self.snapshot(0), self.snapshot(1))
        self.assertTrue(result["all_changed_fields_match_baseline"])
        self.assertIsNone(result["official_success"])

    def test_absent_observable_change_is_unknown(self):
        result = review(self.snapshot(1), self.snapshot(1), self.snapshot(1))
        self.assertIsNone(result["all_changed_fields_match_baseline"])

    def test_recreated_deleted_service(self):
        baseline = {"app": {"items": [{"kind": "Service", "metadata": {"name": "web", "namespace": "app"}, "spec": {"ports": [{"port": 80}]}}]}}
        result = review(baseline, {"app": {"items": []}}, baseline)
        self.assertTrue(result["all_changed_fields_match_baseline"])
