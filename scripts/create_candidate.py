"""Package a trained EfficientNetB0 model with its exact class order."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--output", type=Path, default=Path("models/model-manifest.json"))
    args = parser.parse_args()
    if not 0 < args.confidence_threshold < 1:
        raise SystemExit("confidence threshold must be between zero and one")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if (not audit.get("dataset_manifest_verified") or not audit.get("human_review_complete")
        or not audit["group_split_verified"] or audit["exact_cross_split_duplicates"]
        or audit["conflicting_label_groups"] or not audit["class_sets_match"]):
        raise SystemExit("Dataset audit did not pass")
    labels = sorted(path.name for path in args.train_dir.iterdir() if path.is_dir())
    if labels != sorted(audit["counts"]["train"]):
        raise SystemExit("Training classes differ from audited classes")
    with zipfile.ZipFile(args.model) as archive:
        config = json.loads(archive.read("config.json"))
    if config["config"]["layers"][-1]["config"]["units"] != len(labels):
        raise SystemExit("Model output size does not match training classes")
    digest = hashlib.sha256()
    with args.model.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest = {
        "version": args.version,
        "release_status": "candidate",
        "architecture": "EfficientNetB0",
        "framework": "TensorFlow/Keras 2.16.2",
        "artifact": args.model.name,
        "sha256": digest.hexdigest(),
        "input_size": [224, 224],
        "resize_interpolation": "nearest",
        "confidence_threshold": args.confidence_threshold,
        "preprocessing": "tensorflow.keras.applications.efficientnet.preprocess_input",
        "dataset_version": audit["dataset_version"],
        "dataset_manifest_sha256": audit["dataset_manifest_sha256"],
        "test_metrics": None,
        "class_names": labels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Created candidate {args.version}: {len(labels)} labels, SHA-256 {digest.hexdigest()}")


if __name__ == "__main__":
    main()
