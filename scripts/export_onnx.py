"""Export the pinned Keras candidate to a small ONNX staging artifact."""

import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.verify_release import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("models/efficientnetb0_best.keras"))
    parser.add_argument("--source-manifest", type=Path, default=Path("models/model-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("models/efficientnetb0_candidate.onnx"))
    parser.add_argument("--manifest-output", type=Path, default=Path("models/edge-manifest.json"))
    args = parser.parse_args()

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if sha256(args.source) != source_manifest["sha256"]:
        raise SystemExit("Pinned source Keras checksum mismatch")

    import tensorflow as tf

    model = tf.keras.models.load_model(args.source, compile=False)
    if model.output_shape[-1] != len(source_manifest["class_names"]):
        raise SystemExit("Keras class count mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tempdir:
        with contextlib.redirect_stdout(io.StringIO()):
            model.export(tempdir)
        result = subprocess.run(
            [sys.executable, "-m", "tf2onnx.convert", "--saved-model", tempdir,
             "--output", str(args.output.resolve()), "--opset", "13"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise SystemExit("ONNX conversion failed:\n" + result.stderr[-3000:])

    manifest = dict(source_manifest)
    manifest.update({
        "framework": "ONNX Runtime 1.23.2 (converted from TensorFlow 2.16.2)",
        "artifact": args.output.name,
        "sha256": sha256(args.output),
        "source_keras_sha256": source_manifest["sha256"],
        "deployment_variant": "onnx-cpu",
    })
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {args.output} ({args.output.stat().st_size} bytes, SHA-256 {manifest['sha256']})")
    print("Candidate status and unverified dataset status are unchanged")


if __name__ == "__main__":
    main()
