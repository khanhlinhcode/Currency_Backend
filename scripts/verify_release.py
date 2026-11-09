"""Validate the model contract without importing TensorFlow."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("models/model-manifest.json"))
    parser.add_argument("--model", type=Path, default=Path("models/efficientnetb0_best.keras"))
    parser.add_argument("--require-artifact", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    labels = manifest["class_names"]
    assert len(labels) == len(set(labels)) and len(labels) >= 2, "Invalid class list"
    assert manifest["input_size"] == [224, 224], "Unexpected input size"
    assert manifest.get("resize_interpolation") == "nearest", "Unexpected resize interpolation"
    assert 0 < manifest["confidence_threshold"] < 1, "Invalid confidence threshold"
    assert len(manifest["sha256"]) == 64, "Invalid checksum"
    assert manifest["release_status"] in {"candidate", "approved"}, "Invalid release status"
    if manifest["release_status"] == "approved":
        assert manifest["dataset_version"] != "legacy-unversioned", "Approved model requires a versioned dataset"
        assert len(manifest.get("dataset_manifest_sha256", "")) == 64, "Approved model requires an audited dataset manifest"
        assert len(manifest.get("quality_gates_sha256", "")) == 64, "Approved model requires a locked quality policy"
        assert manifest["test_metrics"] and manifest["test_metrics"]["test_samples"] > 0, "Approved model requires test metrics"
    if args.require_artifact or args.model.exists():
        assert args.model.exists(), "Model artifact is missing"
        assert sha256(args.model) == manifest["sha256"], "Model checksum mismatch"
        if args.model.suffix == ".keras":
            with zipfile.ZipFile(args.model) as archive:
                config = json.loads(archive.read("config.json"))
            output_units = config["config"]["layers"][-1]["config"]["units"]
            assert output_units == len(labels), f"Model has {output_units} outputs; manifest has {len(labels)} labels"
        elif args.model.suffix != ".onnx":
            raise AssertionError("Unsupported model artifact format")
    print(f"Release contract valid: {manifest['version']}, {len(labels)} classes")


if __name__ == "__main__":
    main()
