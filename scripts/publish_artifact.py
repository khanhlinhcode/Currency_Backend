"""Upload one approved, checksum-verified model to private S3 storage."""

import argparse
import json
from pathlib import Path

from botocore.exceptions import ClientError

from api.config import MODEL_S3_BUCKET, MODEL_S3_KEY
from api.storage import _s3_client
from scripts.verify_release import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("models/model-manifest.json"))
    parser.add_argument("--model", type=Path, default=Path("models/efficientnetb0_best.keras"))
    parser.add_argument("--target", choices=["production", "staging"], default="production")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.target == "production" and manifest.get("release_status") != "approved":
        raise SystemExit("Only an approved model can be published to production")
    if args.target == "staging" and manifest.get("release_status") not in {"candidate", "approved"}:
        raise SystemExit("Staging requires a candidate or approved model")
    if not MODEL_S3_BUCKET or not MODEL_S3_KEY:
        raise SystemExit("Set MODEL_S3_BUCKET (or S3_BUCKET) and MODEL_S3_KEY")
    expected_prefix = "staging/models/" if args.target == "staging" else "models/"
    if not MODEL_S3_KEY.startswith(expected_prefix) or (args.target == "production" and MODEL_S3_KEY.startswith("models/staging/")):
        raise SystemExit(f"MODEL_S3_KEY must start with {expected_prefix} for {args.target}")
    if sha256(args.model) != manifest["sha256"]:
        raise SystemExit("Model checksum mismatch")
    client = _s3_client()
    try:
        existing = client.head_object(Bucket=MODEL_S3_BUCKET, Key=MODEL_S3_KEY)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise
    else:
        if existing.get("Metadata", {}).get("sha256") == manifest["sha256"]:
            print(f"Artifact already exists with matching checksum: s3://{MODEL_S3_BUCKET}/{MODEL_S3_KEY}")
            return
        raise SystemExit("Artifact key already exists with a different checksum; use an immutable versioned key")
    client.upload_file(
        str(args.model), MODEL_S3_BUCKET, MODEL_S3_KEY,
        ExtraArgs={"Metadata": {"sha256": manifest["sha256"], "model-version": manifest["version"]}},
    )
    metadata = client.head_object(Bucket=MODEL_S3_BUCKET, Key=MODEL_S3_KEY)["Metadata"]
    if metadata.get("sha256") != manifest["sha256"]:
        raise SystemExit("Uploaded artifact metadata did not match checksum")
    print(f"Published {manifest['release_status']} model {manifest['version']} to s3://{MODEL_S3_BUCKET}/{MODEL_S3_KEY}")


if __name__ == "__main__":
    main()
