"""Build one immutable, audited dataset v2 snapshot after human review completes."""

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from training.review_queue import generate_review_queue, review_issues


ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def run(script, *args):
    command = [sys.executable, str(ROOT / script), *map(str, args)]
    subprocess.run(command, check=True, cwd=ROOT)


def finalize(raw, review_csv, output_root, version):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", version):
        raise ValueError("Dataset version must use 1–64 letters, digits, dots, dashes or underscores")
    raw = raw.resolve()
    review_csv = review_csv.resolve()
    output_root = output_root.resolve()
    if output_root.is_relative_to(raw):
        raise ValueError("Snapshot output cannot be inside the raw dataset")
    final = output_root / version
    if final.exists():
        raise FileExistsError(f"Snapshot version already exists: {final}")
    if not raw.is_dir() or not review_csv.is_file():
        raise FileNotFoundError("Raw dataset or review CSV is missing")
    with review_csv.open(newline="", encoding="utf-8") as source:
        pending = sum(bool(review_issues(row)) for row in csv.DictReader(source))
    if pending:
        raise ValueError(f"Human review incomplete: {pending} images still pending")
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".pending-{version}-", dir=output_root) as name:
        stage = Path(name)
        snapshot_review = stage / "review.csv"
        shutil.copy2(review_csv, snapshot_review)
        review_sha = file_hash(snapshot_review)
        status = generate_review_queue(snapshot_review, stage / "pending.csv", stage / "review-status.json")
        if status["pending_rows"]:
            raise ValueError(f"Human review incomplete: {status['pending_rows']} images still pending")
        run("training/curate_raw.py", "--raw", raw, "--review-csv", snapshot_review,
            "--output", stage / "curated", "--group-template", stage / "groups.csv",
            "--report", stage / "curation.json")
        run("training/split_grouped.py", "--raw", stage / "curated", "--groups", stage / "groups.csv",
            "--dataset-version", version, "--output", stage / "processed",
            "--curation-report", stage / "curation.json", "--attested-physical-groups")
        run("training/audit_dataset.py", "--root", stage / "processed", "--output", stage / "audit.json")
        audit = json.loads((stage / "audit.json").read_text(encoding="utf-8"))
        if not (audit["dataset_manifest_verified"] and audit["human_review_complete"]
                and audit["group_split_verified"] and not audit["duplicate_groups"]
                and not audit["conflicting_label_groups"]):
            raise ValueError("Dataset audit did not pass all release prerequisites")
        if file_hash(review_csv) != review_sha:
            raise ValueError("Review CSV changed during snapshot creation; retry with a new version")
        files = ["review.csv", "groups.csv", "curation.json", "audit.json",
                 "processed/split-manifest.json", "processed/dataset-manifest.json"]
        manifest = {"schema_version": 1, "version": version,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "raw_image_count": status["review_rows"],
                    "dataset_manifest_sha256": audit["dataset_manifest_sha256"],
                    "files": {path: file_hash(stage / path) for path in files},
                    "note": "Human group attestation is recorded per image; hashes detect later edits"}
        (stage / "snapshot.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if final.exists():
            raise FileExistsError(f"Snapshot version appeared during processing: {final}")
        stage.rename(final)
        print(f"Audited dataset v2 snapshot: {final}")
        return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--review-csv", type=Path, default=Path("results/group-review.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset/v2"))
    parser.add_argument("--version")
    args = parser.parse_args()
    version = args.version
    if not version:
        if not sys.stdin.isatty():
            parser.error("--version is required when not running interactively")
        version = input("Dataset version (e.g. 2.0.0): ").strip()
    try:
        finalize(args.raw, args.review_csv, args.output_root, version)
    except (ValueError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Cannot finalize dataset: {error}\n")


if __name__ == "__main__":
    main()
