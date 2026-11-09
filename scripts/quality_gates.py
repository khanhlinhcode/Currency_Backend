"""Validate a pre-registered model quality policy before held-out evaluation."""

import hashlib
import json
from datetime import datetime


METRICS = ("min_accuracy", "min_macro_f1", "min_class_recall", "max_ece", "min_accepted_accuracy", "max_p95_ms")


def load_gates(path, dataset_version, *, require_locked=True):
    content = path.read_bytes()
    gates = json.loads(content)
    if gates.get("schema_version") != 1 or gates.get("dataset_version") != dataset_version:
        raise ValueError("Quality policy schema or dataset version mismatch")
    thresholds = gates.get("thresholds", {})
    if set(thresholds) != set(METRICS):
        raise ValueError("Quality policy must define every required threshold")
    for name in METRICS:
        value = thresholds[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Invalid quality threshold: {name}")
        if name == "max_p95_ms":
            if value <= 0:
                raise ValueError("max_p95_ms must be positive")
        elif not 0 <= value <= 1:
            raise ValueError(f"Quality threshold out of range: {name}")
    if require_locked:
        if gates.get("status") != "locked" or not gates.get("approved_by"):
            raise ValueError("Quality policy must be approved and locked before evaluation")
        try:
            locked_at = datetime.fromisoformat(gates["locked_at"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            raise ValueError("Missing valid policy lock time") from None
        if locked_at.tzinfo is None:
            raise ValueError("Policy lock time must include timezone")
    return gates, hashlib.sha256(content).hexdigest()


def evaluate_checks(manifest, audit, evaluation, benchmark, gates, gates_sha256):
    thresholds = gates["thresholds"]
    try:
        lock_time = datetime.fromisoformat(gates["locked_at"].replace("Z", "+00:00"))
        evaluation_time = datetime.fromisoformat(evaluation["evaluated_at"].replace("Z", "+00:00"))
        policy_precedes_test = lock_time < evaluation_time
    except (KeyError, ValueError):
        policy_precedes_test = False
    return {
        "verified dataset manifest": audit.get("dataset_manifest_verified") is True,
        "dataset manifest checksum": bool(audit.get("dataset_manifest_sha256")),
        "human grouping reviewed": audit.get("human_review_complete") is True,
        "group split verified": audit.get("group_split_verified") is True,
        "class sets match": audit.get("class_sets_match") is True,
        "no exact cross-split duplicates": audit.get("exact_cross_split_duplicates") == 0,
        "no conflicting labels": audit.get("conflicting_label_groups") == 0,
        "dataset version matches": audit.get("dataset_version") == gates.get("dataset_version") == evaluation.get("dataset_version") != "legacy-unversioned",
        "model version matches": evaluation.get("model_version") == manifest.get("version"),
        "model checksum matches": evaluation.get("model_sha256") == manifest.get("sha256"),
        "quality policy locked before test": policy_precedes_test and evaluation.get("gates_sha256") == gates_sha256,
        "test accuracy": evaluation.get("accuracy", 0) >= thresholds["min_accuracy"],
        "test macro F1": evaluation.get("macro_f1", 0) >= thresholds["min_macro_f1"],
        "per-class recall": all(
            evaluation.get("per_class", {}).get(label, {}).get("recall", 0) >= thresholds["min_class_recall"]
            and evaluation.get("per_class", {}).get(label, {}).get("support", 0) > 0
            for label in manifest.get("class_names", [])
        ) and bool(manifest.get("class_names")),
        "confidence calibration": evaluation.get("ece_10_bins", 1) <= thresholds["max_ece"],
        "accepted accuracy": evaluation.get("accepted_accuracy") is not None and evaluation["accepted_accuracy"] >= thresholds["min_accepted_accuracy"],
        "benchmark sample size": benchmark.get("runs", 0) >= 30,
        "benchmark model version": benchmark.get("model_version") == manifest.get("version"),
        "benchmark p95": benchmark.get("p95_ms", float("inf")) <= thresholds["max_p95_ms"],
    }
