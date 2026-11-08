"""Archive an immutable MLOps run folder in private S3 compatible storage."""

import argparse
import json
from pathlib import Path

from botocore.exceptions import ClientError

from api.config import S3_BUCKET
from api.storage import _s3_client
from scripts.verify_release import sha256


def put_immutable(client, key, path):
    digest = sha256(path)
    try:
        existing = client.head_object(Bucket=S3_BUCKET, Key=key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise
    else:
        if existing.get("Metadata", {}).get("sha256") == digest:
            return "already_present"
        raise RuntimeError(f"Archive object differs: {key}")
    client.upload_file(str(path), S3_BUCKET, key, ExtraArgs={"Metadata": {"sha256": digest}})
    saved = client.head_object(Bucket=S3_BUCKET, Key=key)
    if saved.get("Metadata", {}).get("sha256") != digest:
        raise RuntimeError(f"Archived checksum metadata mismatch: {key}")
    return "uploaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--target", choices=["staging", "production"], default="staging")
    args = parser.parse_args()
    record_path = args.run_dir / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("run_id") != args.run_dir.name:
        raise SystemExit("Run ID does not match folder name")
    if args.target == "production" and record.get("status") != "approved":
        raise SystemExit("Only approved runs can be archived under production")
    if not S3_BUCKET:
        raise SystemExit("Set S3_BUCKET")
    for item in record["files"].values():
        path = args.run_dir / item["file"]
        if path.is_symlink() or path.parent != args.run_dir or sha256(path) != item["sha256"]:
            raise SystemExit(f"Run file missing or changed: {path}")
    client = _s3_client()
    prefix = f"{args.target}/runs/{record['run_id']}/"
    for path in sorted(args.run_dir.iterdir()):
        if path.is_file() and not path.is_symlink() and path.name != "run.json":
            put_immutable(client, prefix + path.name, path)
    put_immutable(client, prefix + "run.json", record_path)
    print(f"Archived {record['status']} run {record['run_id']} to s3://{S3_BUCKET}/{prefix}")


if __name__ == "__main__":
    main()
