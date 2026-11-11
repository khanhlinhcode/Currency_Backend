import json
import unittest
from pathlib import Path

from scripts.device_updater import validate_contract, validate_release_contract


ROOT = Path(__file__).resolve().parents[1]


class DeviceContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads((ROOT / "docs/device-contract.staging.example.json").read_text())

    def test_local_contract_requires_complete_unique_classes(self):
        validate_contract(self.contract)
        for broken in ({**self.contract, "class_names": []},
                       {**self.contract, "class_names": ["INR_10", "INR_10"]},
                       {**self.contract, "input_size": [True, 224]},
                       {**self.contract, "channel": "production-preview"},
                       {**self.contract, "schema_version": 2}):
            with self.assertRaises(ValueError):
                validate_contract(broken)

    def test_signed_empty_channel_still_must_match_pinned_channel(self):
        validate_release_contract({"status": "no_release", "channel": "staging"}, self.contract)
        with self.assertRaisesRegex(ValueError, "channel"):
            validate_release_contract({"status": "no_release", "channel": "production"}, self.contract)

    def test_candidate_or_incompatible_manifest_is_rejected(self):
        manifest = {**{key: self.contract[key] for key in (
            "class_names", "input_size", "preprocessing", "resize_interpolation",
            "confidence_threshold")},
            "release_status": "approved", "sha256": "a" * 64,
            "version": "2.0.0", "dataset_version": "2.0.0"}
        release = {"status": "available", "channel": "staging", "model_filename": "model.onnx",
                   "model_sha256": "a" * 64, "model_version": "2.0.0",
                   "dataset_version": "2.0.0", "manifest": manifest}
        validate_release_contract(release, self.contract)
        for broken in ({**release, "manifest": {**manifest, "release_status": "candidate"}},
                       {**release, "manifest": {**manifest, "dataset_version": "legacy-unversioned"}},
                       {**release, "model_filename": "model.keras"},
                       {**release, "manifest": {**manifest, "class_names": list(reversed(manifest["class_names"]))}}):
            with self.assertRaisesRegex(ValueError, "pinned device contract"):
                validate_release_contract(broken, self.contract)


if __name__ == "__main__":
    unittest.main()
