"""Merge verified capture intakes into one new, auditable raw/review snapshot.

No group IDs are inferred or renamed. Cross-campaign group ID, source-path or
exact-content reuse is rejected because physical-note identity is unknown.
"""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from training.import_review_batch import atomic_csv, read_csv
from training.intake_capture import FORMATS, verify_intake
from training.review_app import FIELDS, digest_file


def _canonical_sha256(document):
    canonical = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def verify_merge(snapshot):
    snapshot = Path(snapshot).resolve(strict=True)
    manifest = json.loads((snapshot / "merge-manifest.json").read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_sha256", None)
    if claimed != _canonical_sha256(manifest):
        raise ValueError("Merge manifest hash mismatch")
    review_csv = snapshot / "review.csv"
    if digest_file(review_csv) != manifest.get("review_csv_sha256"):
        raise ValueError("Merged review CSV hash mismatch")
    review_rows = read_csv(review_csv, FIELDS)
    indexed = {row["path"]: row for row in review_rows}
    records = manifest.get("images")
    if not isinstance(records, list) or len(indexed) != len(review_rows) or len(records) != len(review_rows):
        raise ValueError("Merged image/review counts disagree")
    summary = manifest.get("summary")
    intakes = manifest.get("intakes")
    if (not isinstance(summary, dict) or summary.get("images") != len(records)
            or not isinstance(intakes, list) or summary.get("intake_snapshots") != len(intakes)):
        raise ValueError("Merged summary count disagrees")
    intake_hashes = {item["batch_id"]: item["intake_manifest_sha256"] for item in intakes}
    if len(intake_hashes) != len(intakes):
        raise ValueError("Duplicate batch ID in merge manifest")
    seen = set()
    source_paths = set()
    batch_groups = defaultdict(set)
    batch_hashes = defaultdict(set)
    group_labels = {}
    for record in records:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 3:
            raise ValueError("Unsafe merged image path")
        if record["path"] in seen:
            raise ValueError("Duplicate merged image path")
        seen.add(record["path"])
        source_path = record["source_path"]
        if source_path in source_paths:
            raise ValueError(f"Duplicate source path across campaigns: {source_path}")
        source_paths.add(source_path)
        group = record["group_id"]
        batch = record["batch_id"]
        digest = record["sha256"]
        if (not group or not record["source_ref"] or not record["reviewed_by"]
                or batch not in intake_hashes
                or record["intake_manifest_sha256"] != intake_hashes[batch]):
            raise ValueError(f"Missing or inconsistent merged provenance: {relative}")
        if group in group_labels and group_labels[group] != record["label"]:
            raise ValueError(f"Merged group spans classes: {group}")
        group_labels[group] = record["label"]
        batch_groups[group].add(batch)
        batch_hashes[digest].add(batch)
        image = (snapshot / "raw" / relative).resolve(strict=True)
        if not image.is_relative_to(snapshot / "raw") or digest_file(image) != digest:
            raise ValueError(f"Merged image changed: {relative}")
        row = indexed.get(record["path"])
        if row is None or any((row["current_label"] != record["label"],
                               row["reviewed_label"] != record["label"],
                               row["exact_hash_group"] != digest,
                               row["group_id"] != group,
                               row["reviewed_by"] != record["reviewed_by"],
                               row["reviewed_at"] != record["reviewed_at"],
                               row["review_notes"] != record["merged_review_notes"] or
                               row["label_conflict"] != "")):
            raise ValueError(f"Merged review differs from manifest: {relative}")
    if any(len(batches) > 1 for batches in batch_groups.values()):
        raise ValueError("group_id collision across campaigns")
    if any(len(batches) > 1 for batches in batch_hashes.values()):
        raise ValueError("Exact image content reused across campaigns")
    actual = {str(path.relative_to(snapshot / "raw")) for path in (snapshot / "raw").rglob("*")
              if path.is_file() and path.suffix.lower() in FORMATS}
    if actual != seen:
        raise ValueError("Merged raw image coverage differs from manifest")
    return {"verified": True, "merge_id": manifest["merge_id"],
            "images": len(records), "manifest_sha256": claimed}


def merge_intakes(intakes, output, merge_id, apply=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", merge_id):
        raise ValueError("merge-id must be 1–64 letters, digits, dots, dashes or underscores")
    if len(intakes) < 2:
        raise ValueError("At least two intake snapshots are required")
    paths = [Path(path).resolve(strict=True) for path in intakes]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate intake snapshot path")
    output = Path(output).resolve()
    if output.exists() or any(output.is_relative_to(path) or path.is_relative_to(output) for path in paths):
        raise ValueError("Merge output must be new and separate from all intakes")
    snapshots = []
    snapshot_sources = {}
    batch_ids = set()
    source_paths = set()
    group_batches = defaultdict(set)
    digest_batches = defaultdict(set)
    merged_records = []
    for path in paths:
        verified = verify_intake(path)
        batch_id = verified["batch_id"]
        if batch_id in batch_ids:
            raise ValueError(f"Duplicate intake batch_id: {batch_id}")
        batch_ids.add(batch_id)
        snapshot_sources[batch_id] = path
        intake_manifest = json.loads((path / "intake-manifest.json").read_text(encoding="utf-8"))
        review_rows = {row["path"]: row for row in read_csv(path / "review.csv", FIELDS)}
        snapshots.append({"batch_id": batch_id,
                          "intake_manifest_sha256": verified["manifest_sha256"],
                          "review_csv_sha256": intake_manifest["review_csv_sha256"],
                          "images": verified["images"]})
        for record in intake_manifest["images"]:
            source_path = record["source_path"]
            if source_path in source_paths:
                raise ValueError(f"Duplicate source_path across campaigns: {source_path}")
            source_paths.add(source_path)
            group = record["group_id"]
            digest = record["sha256"]
            group_batches[group].add(batch_id)
            digest_batches[digest].add(batch_id)
            original = Path(record["path"])
            target_name = hashlib.sha256(f"{batch_id}:{record['path']}".encode()).hexdigest()[:20] + original.suffix.lower()
            target = Path(original.parts[0]) / original.parts[1] / target_name
            review = review_rows[record["path"]]
            note = f"Intake batch: {batch_id}; {review['review_notes']}"
            if len(note) > 2000:
                raise ValueError(f"Merged review note too long: {source_path}")
            merged_records.append({"path": str(target), "source_path": source_path,
                                   "intake_path": record["path"], "batch_id": batch_id,
                                   "intake_manifest_sha256": verified["manifest_sha256"],
                                   "sha256": digest, "label": record["label"],
                                   "group_id": group, "source_ref": record["source_ref"],
                                   "captured_by": record["captured_by"],
                                   "captured_at": record["captured_at"],
                                   "reviewed_by": record["reviewed_by"],
                                   "reviewed_at": record["reviewed_at"],
                                   "merged_review_notes": note})
    if any(len(batches) > 1 for batches in group_batches.values()):
        raise ValueError("group_id collision across campaigns; attest globally unique IDs at capture")
    if any(len(batches) > 1 for batches in digest_batches.values()):
        raise ValueError("Exact image content reused across campaigns")
    if len({row["path"] for row in merged_records}) != len(merged_records):
        raise ValueError("Merged destination path collision")
    snapshots.sort(key=lambda item: item["batch_id"])
    group_labels = {}
    for record in merged_records:
        group = record["group_id"]
        if group in group_labels and group_labels[group] != record["label"]:
            raise ValueError(f"group_id spans classes: {group}")
        group_labels[group] = record["label"]
    counts = Counter(row["label"] for row in merged_records)
    groups_by_class = defaultdict(set)
    for record in merged_records:
        groups_by_class[record["label"]].add(record["group_id"])
    summary = {"schema_version": 1, "merge_id": merge_id, "mode": "apply" if apply else "check",
               "intake_snapshots": len(paths), "images": len(merged_records),
               "classes": dict(sorted(counts.items())),
               "groups_per_class": {label: len(groups) for label, groups in sorted(groups_by_class.items())},
               "classes_below_three_groups": sorted(label for label, groups in groups_by_class.items() if len(groups) < 3),
               "cross_campaign_group_collisions": 0, "cross_campaign_exact_duplicates": 0}
    if not apply:
        return summary
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".merge-{merge_id}-", dir=output.parent) as temporary:
        stage = Path(temporary)
        by_batch = snapshot_sources
        review_rows = []
        for row in sorted(merged_records, key=lambda item: item["path"]):
            source = by_batch[row["batch_id"]] / "raw" / row["intake_path"]
            destination = stage / "raw" / row["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.parent.chmod(0o700)
            shutil.copyfile(source, destination)
            destination.chmod(0o600)
            if digest_file(destination) != row["sha256"]:
                raise ValueError(f"Image changed during merge: {row['source_path']}")
            review_rows.append({"path": row["path"], "current_label": row["label"],
                                "exact_hash_group": row["sha256"], "label_conflict": "",
                                "reviewed_label": row["label"], "group_id": row["group_id"],
                                "reviewed_by": row["reviewed_by"], "reviewed_at": row["reviewed_at"],
                                "review_notes": row["merged_review_notes"]})
        review_csv = stage / "review.csv"
        atomic_csv(review_csv, review_rows, FIELDS)
        manifest = {"schema_version": 1, "merge_id": merge_id,
                    "summary": summary, "intakes": snapshots,
                    "review_csv_sha256": digest_file(review_csv),
                    "images": sorted(merged_records, key=lambda item: item["path"])}
        manifest["manifest_sha256"] = _canonical_sha256(manifest)
        manifest_path = stage / "merge-manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest_path.chmod(0o600)
        (stage / "raw").chmod(0o700)
        verify_merge(stage)
        for snapshot in snapshots:
            path = snapshot_sources[snapshot["batch_id"]]
            current = verify_intake(path)
            if current["manifest_sha256"] != snapshot["intake_manifest_sha256"]:
                raise ValueError("Intake changed during merge")
        if output.exists():
            raise FileExistsError(f"Merge output appeared during staging: {output}")
        stage.rename(output)
    return {**summary, "output": str(output), "manifest_sha256": manifest["manifest_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, action="append", help="Repeat for each intake snapshot")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--merge-id")
    parser.add_argument("--verify-merge", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_merge:
            if args.intake or args.output or args.merge_id or args.apply:
                parser.error("--verify-merge cannot be combined with merge options")
            result = verify_merge(args.verify_merge)
        else:
            if not args.intake or not args.output or not args.merge_id:
                parser.error("Supply at least two --intake paths, --output and --merge-id")
            result = merge_intakes(args.intake, args.output, args.merge_id, apply=args.apply)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Capture merge rejected: {error}\n")


if __name__ == "__main__":
    main()
