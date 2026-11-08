"""Evaluate an ONNX runtime artifact on the historical test split (diagnostic)."""

import argparse
import json
import math
import platform
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from scripts.verify_release import sha256


def percentile(values, q):
    ordered = sorted(values)
    return round(ordered[math.ceil(q * len(ordered)) - 1], 3)


def metrics(rows, labels, threshold):
    total = len(rows)
    correct = sum(actual == predicted for actual, predicted, _ in rows)
    accepted = [(a, p) for a, p, confidence in rows if confidence >= threshold]
    class_metrics = {}
    for label in labels:
        tp = sum(a == label and p == label for a, p, _ in rows)
        support = sum(a == label for a, _, _ in rows)
        predicted_count = sum(p == label for _, p, _ in rows)
        precision = tp / predicted_count if predicted_count else 0
        recall = tp / support if support else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        class_metrics[label] = {"support": support, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}
    ece = 0
    for bin_index in range(10):
        low, high = bin_index / 10, (bin_index + 1) / 10
        subset = [(a, p, c) for a, p, c in rows if low <= c < high or (bin_index == 9 and c == 1)]
        if subset:
            bin_accuracy = sum(a == p for a, p, _ in subset) / len(subset)
            bin_confidence = statistics.mean(c for _, _, c in subset)
            ece += len(subset) / total * abs(bin_accuracy - bin_confidence)
    return {
        "test_samples": total,
        "accuracy": round(correct / total, 6),
        "macro_f1": round(statistics.mean(value["f1"] for value in class_metrics.values()), 6),
        "ece_10_bins": round(ece, 6),
        "accepted_coverage": round(len(accepted) / total, 6),
        "accepted_accuracy": round(sum(a == p for a, p in accepted) / len(accepted), 6) if accepted else None,
        "high_confidence_errors": sum(a != p for a, p in accepted),
        "per_class": class_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--model", type=Path, default=Path("models/efficientnetb0_candidate.onnx"))
    parser.add_argument("--manifest", type=Path, default=Path("models/edge-manifest.json"))
    parser.add_argument("--audit", type=Path, default=Path("results/dataset-audit.json"))
    parser.add_argument("--output", type=Path, default=Path("results/onnx-evaluation.json"))
    parser.add_argument("--compare-source", type=Path, help="Compare predictions to source Keras on identical resized images")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if sha256(args.model) != manifest["sha256"]:
        raise SystemExit("ONNX artifact checksum mismatch")
    labels = manifest["class_names"]
    test_dir = args.dataset_root / "test"
    images = [(directory.name.replace("USA_2 Doolar", "USA_2 Dollar"), path)
              for directory in sorted(test_dir.iterdir()) if directory.is_dir()
              for path in sorted(directory.rglob("*"))
              if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not images:
        raise SystemExit("Test split is empty")
    if set(label for label, _ in images) != set(labels):
        raise SystemExit("Test labels do not match the model manifest")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    rows, times = [], []
    source_rows, agreement, probability_differences = [], 0, []
    source_model = None
    if args.compare_source:
        if sha256(args.compare_source) != manifest["source_keras_sha256"]:
            raise SystemExit("Source Keras artifact checksum mismatch")
        import tensorflow as tf
        source_model = tf.keras.models.load_model(args.compare_source, compile=False)
    started = time.perf_counter()
    for label, path in images:
        with Image.open(path) as source:
            tensor = np.expand_dims(np.asarray(source.convert("RGB").resize((224, 224), Image.Resampling.NEAREST), dtype=np.float32), 0)
        inference_started = time.perf_counter()
        scores = session.run([output_name], {input_name: tensor})[0][0]
        times.append((time.perf_counter() - inference_started) * 1000)
        rows.append((label, labels[int(np.argmax(scores))], float(np.max(scores))))
        if source_model is not None:
            source_scores = source_model(tensor, training=False).numpy()[0]
            source_rows.append((label, labels[int(np.argmax(source_scores))], float(np.max(source_scores))))
            agreement += int(np.argmax(scores) == np.argmax(source_scores))
            probability_differences.append(float(np.max(np.abs(scores - source_scores))))
    elapsed = time.perf_counter() - started

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "interpretation": "Diagnostic only: historical split has leakage and unverified physical groups",
        "model_version": manifest["version"],
        "model_sha256": manifest["sha256"],
        "onnx_sha256": manifest["sha256"],
        "source_keras_sha256": manifest["source_keras_sha256"],
        "dataset_version": audit["dataset_version"],
        "group_split_verified": bool(audit["group_split_verified"]),
        "exact_cross_split_duplicates": audit["exact_cross_split_duplicates"],
        "conflicting_label_groups": audit["conflicting_label_groups"],
        "confidence_threshold": manifest["confidence_threshold"],
        **metrics(rows, labels, manifest["confidence_threshold"]),
        "wall_seconds": round(elapsed, 3),
        "images_per_second": round(len(rows) / elapsed, 3),
        "inference_ms": {"p50": percentile(times, .5), "p95": percentile(times, .95), "p99": percentile(times, .99)},
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "onnxruntime": ort.__version__},
        "label_counts": dict(Counter(label for label, _ in images)),
    }
    if source_model is not None:
        report["source_same_preprocessing"] = metrics(source_rows, labels, manifest["confidence_threshold"])
        report["source_top1_agreement"] = round(agreement / len(rows), 6)
        report["max_probability_abs_difference"] = max(probability_differences)
        report["mean_probability_max_abs_difference"] = statistics.mean(probability_differences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "test_samples", "accuracy", "macro_f1", "ece_10_bins", "accepted_coverage", "accepted_accuracy",
        "inference_ms", "images_per_second", "group_split_verified")}, indent=2))


if __name__ == "__main__":
    main()
