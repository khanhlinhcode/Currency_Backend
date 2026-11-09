"""Create reproducible test metrics for one pinned model and dataset split."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("models/model-manifest.json"))
    parser.add_argument("--model", type=Path, default=Path("models/efficientnetb0_best.keras"))
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/processed/test"))
    parser.add_argument("--output", type=Path, default=Path("results/release-evaluation.json"))
    parser.add_argument("--audit", type=Path, help="Optional dataset audit for exact-duplicate exclusion diagnostic")
    parser.add_argument("--gates-file", type=Path, help="Locked quality policy registered before this test")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    gates_sha256 = None
    if args.gates_file:
        gates_bytes = args.gates_file.read_bytes()
        gates = json.loads(gates_bytes)
        if (gates.get("status") != "locked" or not gates.get("approved_by")
            or gates.get("dataset_version") != manifest["dataset_version"]):
            raise SystemExit("A locked quality policy for this dataset is required")
        locked_at = datetime.fromisoformat(gates["locked_at"].replace("Z", "+00:00"))
        if locked_at.tzinfo is None or locked_at >= datetime.now(timezone.utc):
            raise SystemExit("Quality policy must predate evaluation")
        gates_sha256 = hashlib.sha256(gates_bytes).hexdigest()
    digest = hashlib.sha256()
    with args.model.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == manifest["sha256"], "Model checksum mismatch"
    canonical = manifest["class_names"]
    label_to_index = {label: index for index, label in enumerate(canonical)}
    generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        args.test_dir, target_size=tuple(manifest["input_size"]), batch_size=32,
        class_mode="categorical", shuffle=False, interpolation=manifest.get("resize_interpolation", "nearest"),
    )
    discovered = {name.replace("USA_2 Doolar", "USA_2 Dollar"): index for name, index in generator.class_indices.items()}
    assert set(discovered) == set(canonical), "Test labels differ from model manifest"
    index_to_canonical = {index: label_to_index[name] for name, index in discovered.items()}
    y_true = np.array([index_to_canonical[int(index)] for index in generator.classes])
    model = tf.keras.models.load_model(args.model, compile=False)
    assert model.output_shape[-1] == len(canonical), "Model output mismatch"
    started = time.perf_counter()
    probabilities = model.predict(generator, verbose=1)
    elapsed = time.perf_counter() - started
    y_pred = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = (y_true == y_pred)
    threshold = manifest["confidence_threshold"]
    accepted = confidence >= threshold
    calibration_error = 0.0
    edges = np.linspace(0, 1, 11)
    for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (confidence >= lower) & (confidence <= upper if bin_index == 9 else confidence < upper)
        if in_bin.any():
            calibration_error += float(in_bin.mean()) * abs(float(correct[in_bin].mean()) - float(confidence[in_bin].mean()))
    report = classification_report(
        y_true, y_pred, labels=list(range(len(canonical))), target_names=canonical,
        output_dict=True, zero_division=0,
    )
    output = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "gates_sha256": gates_sha256,
        "model_version": manifest["version"],
        "model_sha256": manifest["sha256"],
        "dataset_version": manifest["dataset_version"],
        "test_samples": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "ece_10_bins": round(calibration_error, 6),
        "confidence_threshold": threshold,
        "accepted_coverage": round(float(accepted.mean()), 6),
        "accepted_accuracy": round(float(correct[accepted].mean()), 6) if accepted.any() else None,
        "high_confidence_errors": int((accepted & ~correct).sum()),
        "evaluation_seconds": round(elapsed, 3),
        "images_per_second": round(len(y_true) / elapsed, 3),
        "per_class": {label: report[label] for label in canonical},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(canonical)))).tolist(),
    }
    if args.audit:
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
        leaked_test_paths = {
            Path(item["path"]).resolve()
            for group in audit["duplicate_groups"]
            for item in group if item["split"] == "test"
        }
        keep = np.array([Path(path).resolve() not in leaked_test_paths for path in generator.filepaths])
        if not keep.any():
            raise RuntimeError("No test images remain after duplicate exclusion")
        output["exact_duplicate_exclusion"] = {
            "audit_dataset_version": audit["dataset_version"],
            "excluded_test_images": int((~keep).sum()),
            "remaining_test_images": int(keep.sum()),
            "accuracy": round(float(accuracy_score(y_true[keep], y_pred[keep])), 6),
            "macro_f1": round(float(f1_score(y_true[keep], y_pred[keep], labels=list(range(len(canonical))), average="macro", zero_division=0)), 6),
            "note": "Diagnostic only; near-duplicates and repeated physical notes may remain",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Accuracy={output['accuracy']:.4f} macro-F1={output['macro_f1']:.4f} images/s={output['images_per_second']:.1f}")


if __name__ == "__main__":
    main()
