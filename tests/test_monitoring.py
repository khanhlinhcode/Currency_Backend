import unittest

from api.monitoring import assess


TARGETS = {
    "minimum_model_samples": 3, "minimum_reviewed_samples": 2,
    "warning_processing_p95_ms": 2500, "critical_processing_p95_ms": 5000,
    "warning_api_5xx_24h": 1, "critical_api_5xx_24h": 5,
    "max_low_confidence_increase": 0.1, "warning_distribution_js": 0.15,
    "warning_storage_bytes": 800000000,
}


class MonitoringTest(unittest.TestCase):
    def test_no_baseline_never_reports_healthy_model(self):
        stats = {"processing_latency": {"p95_ms": None, "p99_ms": None},
                 "inference_latency": {"p95_ms": None, "p99_ms": None},
                 "reviewed_count": 0, "reviewed_accuracy": None}
        result = assess(stats, [], None, 0, 100, TARGETS)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["drift_status"], "insufficient_data")
        self.assertIsNone(result["reviewed_accuracy"])

    def test_error_and_drift_thresholds(self):
        stats = {"processing_latency": {"p95_ms": 1000, "p99_ms": 1500},
                 "inference_latency": {"p95_ms": 300, "p99_ms": 400},
                 "reviewed_count": 2, "reviewed_accuracy": 0.5}
        predictions = [{"prediction": "B", "confidence": 0.4, "threshold": 0.6} for _ in range(3)]
        baseline = {"class_counts": {"A": 3}, "accepted_coverage": 0.95}
        result = assess(stats, predictions, baseline, 0, 100, TARGETS)
        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["drift_status"], "warning")
        self.assertGreater(result["distribution_js_divergence"], 0.9)
        self.assertEqual(result["reviewed_accuracy"], 0.5)
        self.assertEqual(assess(stats, predictions, baseline, 5, 100, TARGETS)["status"], "critical")


if __name__ == "__main__":
    unittest.main()
