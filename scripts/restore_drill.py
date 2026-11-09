"""Verify an encrypted staging backup by restoring only to a local isolated PostgreSQL database."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from sqlalchemy.engine import make_url

from scripts.backup_staging import MAGIC, database_environment, encryption_key, sha256


def decrypt_file(source, destination, key):
    with source.open("rb") as encrypted:
        if encrypted.read(len(MAGIC)) != MAGIC:
            raise ValueError("Unsupported backup format")
        nonce = encrypted.read(12)
        encrypted.seek(-16, os.SEEK_END)
        tag = encrypted.read(16)
        remaining = source.stat().st_size - len(MAGIC) - 12 - 16
        encrypted.seek(len(MAGIC) + 12)
        decryptor = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()
        with destination.open("xb") as plain:
            destination.chmod(0o600)
            while remaining:
                chunk = encrypted.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("Truncated backup")
                remaining -= len(chunk)
                plain.write(decryptor.update(chunk))
            plain.write(decryptor.finalize())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--restore-database-url", required=True)
    parser.add_argument("--confirm-isolated-target", action="store_true")
    args = parser.parse_args()
    url = make_url(args.restore_database_url)
    if (not args.confirm_isolated_target or url.get_backend_name() != "postgresql"
        or url.host not in {"localhost", "127.0.0.1", "::1"}):
        parser.error("Restore target must be an explicitly confirmed local PostgreSQL database")
    if not shutil.which("pg_restore"):
        parser.error("pg_restore must be installed on the restore client")
    manifest = json.loads((args.backup_dir / "backup-manifest.json").read_text(encoding="utf-8"))
    for filename, item in manifest["encrypted_files"].items():
        if sha256(args.backup_dir / filename) != item["sha256"]:
            raise SystemExit(f"Encrypted backup hash mismatch: {filename}")
    key = encryption_key()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="currency-restore-drill-") as temporary:
        temp = Path(temporary)
        dump = temp / "database.dump"
        objects = temp / "objects.json"
        decrypt_file(args.backup_dir / "database.dump.enc", dump, key)
        decrypt_file(args.backup_dir / "objects.json.enc", objects, key)
        object_manifest = json.loads(objects.read_text(encoding="utf-8"))
        if len(object_manifest["objects"]) != manifest["object_count"]:
            raise SystemExit("Object manifest count mismatch")
        subprocess.run(["pg_restore", "--clean", "--if-exists", "--no-owner", "--no-acl",
                        "--exit-on-error", "--dbname", url.database, str(dump)],
                       env=database_environment(args.restore_database_url), check=True)
    with psycopg.connect(args.restore_database_url.replace("postgresql+psycopg://", "postgresql://")) as connection:
        tables = ("prediction_history", "users", "mlops_runs", "mlops_benchmarks", "mlops_monitor_snapshots")
        counts = {}
        for table in tables:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT count(*) FROM {table}")
                counts[table] = cursor.fetchone()[0]
    report = {"backup_id": manifest["backup_id"], "restored_at": datetime.now(timezone.utc).isoformat(),
              "target": "local-isolated-postgresql", "duration_seconds": round(time.perf_counter() - started, 3),
              "table_counts": counts, "object_manifest_count": manifest["object_count"],
              "encrypted_hashes_verified": True, "restore_ok": True,
              "note": "Object contents were not restored; this drill verifies the database and encrypted object manifest"}
    destination = args.backup_dir / "restore-report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    destination.chmod(0o600)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
