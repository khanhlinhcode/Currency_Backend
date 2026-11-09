"""Create an encrypted PostgreSQL dump and object manifest in a separate private bucket."""

import argparse
import base64
import binascii
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from botocore.exceptions import ClientError
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from sqlalchemy.engine import make_url

from api.config import DATABASE_URL, ENVIRONMENT, S3_BUCKET
from api.storage import _s3_client


MAGIC = b"CURRENCY-AI-AESGCM-V1\n"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def database_environment(url_text):
    url = make_url(url_text)
    if url.get_backend_name() != "postgresql" or not url.host or not url.database:
        raise ValueError("A PostgreSQL DATABASE_URL is required")
    return dict(os.environ, PGHOST=url.host, PGPORT=str(url.port or 5432),
                PGUSER=url.username or "postgres", PGPASSWORD=url.password or "",
                PGDATABASE=url.database, PGSSLMODE=url.query.get("sslmode", "require"))


def encryption_key():
    raw = os.environ.get("BACKUP_KEY_BASE64", "")
    try:
        key = base64.b64decode(raw, validate=True)
    except (ValueError, binascii.Error):
        key = b""
    if len(key) != 32:
        raise ValueError("BACKUP_KEY_BASE64 must contain a 32-byte random key")
    return key


def encrypt_file(source, destination, key):
    nonce = os.urandom(12)
    encryptor = Cipher(algorithms.AES(key), modes.GCM(nonce)).encryptor()
    with source.open("rb") as plain, destination.open("xb") as encrypted:
        encrypted.write(MAGIC + nonce)
        for chunk in iter(lambda: plain.read(1024 * 1024), b""):
            encrypted.write(encryptor.update(chunk))
        encrypted.write(encryptor.finalize())
        encrypted.write(encryptor.tag)
    destination.chmod(0o600)


def put_immutable(client, bucket, key, path):
    digest = sha256(path)
    try:
        existing = client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise
    else:
        if existing.get("Metadata", {}).get("sha256") == digest:
            return
        raise RuntimeError(f"Backup object already exists with different hash: {key}")
    client.upload_file(str(path), bucket, key, ExtraArgs={"Metadata": {"sha256": digest}})
    saved = client.head_object(Bucket=bucket, Key=key)
    if saved.get("Metadata", {}).get("sha256") != digest:
        raise RuntimeError("Backup upload checksum metadata mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Private directory outside the Git checkout")
    parser.add_argument("--backup-bucket", default=os.getenv("BACKUP_S3_BUCKET", ""))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output == root or root in output.parents:
        parser.error("Backup directory must be outside the repository")
    if ENVIRONMENT != "staging" or not S3_BUCKET or not args.backup_bucket or args.backup_bucket == S3_BUCKET:
        parser.error("APP_ENV=staging and a separate private BACKUP_S3_BUCKET are required")
    if not shutil.which("pg_dump"):
        parser.error("pg_dump must be installed on the backup client")
    key = encryption_key()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output / run_id
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    client = _s3_client()
    client.head_bucket(Bucket=args.backup_bucket)
    with tempfile.TemporaryDirectory(prefix="currency-db-backup-") as temporary:
        dump = Path(temporary) / "database.dump"
        with dump.open("wb") as target:
            subprocess.run(["pg_dump", "--format=custom", "--no-owner", "--no-acl"],
                           env=database_environment(DATABASE_URL), stdout=target, check=True)
        encrypt_file(dump, destination / "database.dump.enc", key)
        objects = []
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET):
            for item in page.get("Contents", []):
                objects.append({"key": item["Key"], "size": item["Size"],
                                "etag": item.get("ETag"), "modified_at": item["LastModified"].isoformat()})
        object_manifest = Path(temporary) / "objects.json"
        object_manifest.write_text(json.dumps({"source_bucket": S3_BUCKET,
            "objects": sorted(objects, key=lambda entry: entry["key"])}, ensure_ascii=False), encoding="utf-8")
        encrypt_file(object_manifest, destination / "objects.json.enc", key)
    report = {"schema_version": 1, "backup_id": run_id,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "source_environment": ENVIRONMENT, "object_count": len(objects),
              "encrypted_files": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                                  for path in destination.glob("*.enc")}}
    manifest_path = destination / "backup-manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.chmod(0o600)
    prefix = f"staging/backups/{run_id}/"
    for path in sorted(destination.iterdir()):
        put_immutable(client, args.backup_bucket, prefix + path.name, path)
    print(json.dumps({"backup_id": run_id, "encrypted_files": list(report["encrypted_files"]),
                      "object_count": len(objects), "private_backup_bucket": args.backup_bucket}, ensure_ascii=False))


if __name__ == "__main__":
    main()
