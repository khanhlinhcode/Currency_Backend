"""Validate and stage a newly captured, provenance-attested image batch.

All labels and group IDs come from the capture manifest supplied by people.
Image content is checked for integrity; it is never used to infer a group.
The default mode only checks. --apply creates one new local intake snapshot.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from training.review_app import FIELDS, digest_file
from training.import_review_batch import atomic_csv, read_csv


CAPTURE_FIELDS = ["source_path", "sha256", "label", "group_id", "source_ref",
                  "captured_by", "captured_at", "reviewed_by", "reviewed_at", "review_notes"]
FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
ROOT = Path(__file__).resolve().parents[1]


def _short(value, field, limit, minimum=1):
    value = value.strip()
    if not minimum <= len(value) <= limit or any(ord(char) < 32 for char in value):
        raise ValueError(f"Invalid {field}")
    return value


def _time(value, field):
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be ISO-8601 with timezone") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include timezone")
    return parsed.astimezone(timezone.utc)


def _classes(labels_manifest):
    manifest = json.loads(Path(labels_manifest).read_text(encoding="utf-8"))
    labels = manifest.get("class_names")
    if not isinstance(labels, list) or not labels or len(set(labels)) != len(labels):
        raise ValueError("Invalid class_names in labels manifest")
    return set(labels)


def _review_note(row):
    note = (f"Original source image: {row['source_path']}; source record: {row['source_ref']}; "
            f"captured by {row['captured_by']} at {row['captured_at']}")
    if row["review_notes"]:
        note += f"; review note: {row['review_notes']}"
    return note


def verify_intake(snapshot):
    snapshot = Path(snapshot).resolve(strict=True)
    manifest = json.loads((snapshot / "intake-manifest.json").read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_sha256", None)
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if claimed != hashlib.sha256(canonical).hexdigest():
        raise ValueError("Intake manifest hash mismatch")
    review_csv = snapshot / "review.csv"
    if digest_file(review_csv) != manifest.get("review_csv_sha256"):
        raise ValueError("Intake review CSV hash mismatch")
    review_rows = read_csv(review_csv, FIELDS)
    by_path = {row["path"]: row for row in review_rows}
    records = manifest.get("images")
    if not isinstance(records, list) or len(by_path) != len(review_rows) or len(records) != len(review_rows):
        raise ValueError("Intake image/review counts disagree")
    summary = manifest.get("summary")
    if not isinstance(summary, dict) or summary.get("images") != len(records):
        raise ValueError("Intake summary count differs from manifest")
    seen = set()
    group_labels = {}
    by_digest = defaultdict(list)
    for record in records:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 3:
            raise ValueError("Unsafe image path in intake manifest")
        if record["path"] in seen:
            raise ValueError("Duplicate image path in intake manifest")
        seen.add(record["path"])
        if not record["group_id"] or not record["source_ref"] or not record["reviewed_by"]:
            raise ValueError(f"Missing provenance in intake manifest: {relative}")
        group = record["group_id"]
        if group in group_labels and group_labels[group] != record["label"]:
            raise ValueError(f"Intake group spans classes: {group}")
        group_labels[group] = record["label"]
        by_digest[record["sha256"]].append(record)
        image = (snapshot / "raw" / relative).resolve(strict=True)
        if not image.is_relative_to(snapshot / "raw") or digest_file(image) != record["sha256"]:
            raise ValueError(f"Intake image changed: {relative}")
        row = by_path.get(record["path"])
        if row is None or any((row["current_label"] != record["label"],
                               row["reviewed_label"] != record["label"],
                               row["exact_hash_group"] != record["sha256"],
                               row["group_id"] != record["group_id"],
                               row["reviewed_by"] != record["reviewed_by"],
                               row["reviewed_at"] != record["reviewed_at"],
                               row["review_notes"] != _review_note(record))):
            raise ValueError(f"Intake review differs from manifest: {relative}")
    actual = {str(path.relative_to(snapshot / "raw")) for path in (snapshot / "raw").rglob("*")
              if path.is_file() and path.suffix.lower() in FORMATS}
    if actual != seen:
        raise ValueError("Intake raw image coverage differs from manifest")
    for digest, copies in by_digest.items():
        if len({row["label"] for row in copies}) > 1 or len({row["group_id"] for row in copies}) > 1:
            raise ValueError(f"Exact intake copies disagree on label or group: {digest}")
    return {"verified": True, "batch_id": manifest["batch_id"],
            "images": len(records), "manifest_sha256": claimed}


def export_template(source_root, output):
    source_root = Path(source_root).resolve(strict=True)
    output = Path(output)
    if not source_root.is_dir() or output.exists():
        raise ValueError("source-root must be a directory and template output must be new")
    rows = []
    for image in sorted(source_root.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in FORMATS:
            continue
        if not image.resolve(strict=True).is_relative_to(source_root):
            raise ValueError(f"Image escapes source-root: {image}")
        rows.append({"source_path": str(image.relative_to(source_root)),
                     "sha256": digest_file(image), "label": "", "group_id": "",
                     "source_ref": "", "captured_by": "", "captured_at": "",
                     "reviewed_by": "", "reviewed_at": "", "review_notes": ""})
    if not rows:
        raise ValueError("No supported images in source-root")
    atomic_csv(output, rows, CAPTURE_FIELDS)
    return len(rows)


def validate(source_root, source_csv, labels_manifest):
    source_root = Path(source_root).resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("source-root must be a directory")
    allowed = _classes(labels_manifest)
    with Path(source_csv).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != CAPTURE_FIELDS:
            raise ValueError(f"Capture CSV columns must be: {', '.join(CAPTURE_FIELDS)}")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Capture CSV is empty or has missing/extra cells")

    images = set()
    destinations = set()
    by_digest = defaultdict(list)
    group_labels = {}
    records = []
    for row in rows:
        relative = Path(row["source_path"])
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                or relative.suffix.lower() not in FORMATS):
            raise ValueError(f"Unsafe or unsupported source_path: {relative}")
        path_text = _short(str(relative), "source_path", 500)
        if path_text in images:
            raise ValueError(f"Duplicate source_path: {path_text}")
        images.add(path_text)
        image = (source_root / relative).resolve(strict=True)
        if not image.is_relative_to(source_root) or not image.is_file():
            raise ValueError(f"Source image escapes source-root: {relative}")
        expected = row["sha256"].strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or digest_file(image) != expected:
            raise ValueError(f"Source image hash mismatch: {relative}")
        try:
            with Image.open(image) as opened:
                if opened.format != FORMATS[relative.suffix.lower()]:
                    raise ValueError(f"Image encoding does not match extension: {relative}")
                width, height = opened.size
                if min(width, height) < 32 or width * height > 25_000_000:
                    raise ValueError(f"Unsupported image dimensions: {relative}")
                opened.verify()
            with Image.open(image) as decoded:
                decoded.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError(f"Invalid image: {relative}") from error

        label = _short(row["label"], "label", 100)
        if label not in allowed:
            raise ValueError(f"Unknown class label: {label}")
        country, denomination = label.split("_", 1)
        group = _short(row["group_id"], "group_id", 128)
        if group in group_labels and group_labels[group] != label:
            raise ValueError(f"group_id spans classes: {group}")
        group_labels[group] = label
        reference = _short(row["source_ref"], "source_ref", 300)
        captured_by = _short(row["captured_by"], "captured_by", 80, 2)
        reviewed_by = _short(row["reviewed_by"], "reviewed_by", 80, 2)
        captured_at = _time(row["captured_at"], "captured_at")
        reviewed_at = _time(row["reviewed_at"], "reviewed_at")
        if reviewed_at < captured_at:
            raise ValueError(f"reviewed_at precedes captured_at: {relative}")
        note = row["review_notes"].strip()
        if len(note) > 1000 or any(ord(char) < 32 for char in note):
            raise ValueError(f"Invalid review_notes: {relative}")
        target_name = hashlib.sha256(path_text.encode("utf-8")).hexdigest()[:16] + relative.suffix.lower()
        target = Path(country) / denomination / target_name
        if str(target) in destinations:
            raise ValueError(f"Destination collision: {target}")
        destinations.add(str(target))
        record = {"source_path": path_text, "path": str(target), "sha256": expected,
                  "label": label, "group_id": group, "source_ref": reference,
                  "captured_by": captured_by, "captured_at": captured_at.isoformat(),
                  "reviewed_by": reviewed_by, "reviewed_at": reviewed_at.isoformat(),
                  "review_notes": note, "width": width, "height": height}
        records.append(record)
        by_digest[expected].append(record)

    actual_images = {str(path.relative_to(source_root)) for path in source_root.rglob("*")
                     if path.is_file() and path.suffix.lower() in FORMATS}
    if images != actual_images:
        raise ValueError(f"Capture CSV does not cover source images: {len(actual_images - images)} missing, {len(images - actual_images)} extra")
    for digest, copies in by_digest.items():
        if len({row["label"] for row in copies}) > 1 or len({row["group_id"] for row in copies}) > 1:
            raise ValueError(f"Exact image copies disagree on label or group: {digest}")
    return sorted(records, key=lambda row: row["path"])


def intake(source_root, source_csv, output, batch_id, labels_manifest, apply=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", batch_id):
        raise ValueError("batch-id must be 1–64 letters, digits, dots, dashes or underscores")
    source_root = Path(source_root).resolve(strict=True)
    output = Path(output).resolve()
    if output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError("Output and source-root must be separate")
    if output.exists():
        raise FileExistsError(f"Intake output already exists: {output}")
    source_hash = digest_file(Path(source_csv))
    records = validate(source_root, source_csv, labels_manifest)
    if digest_file(Path(source_csv)) != source_hash:
        raise ValueError("Capture CSV changed during validation")
    summary = {"schema_version": 1, "batch_id": batch_id, "mode": "apply" if apply else "check",
               "source_csv_sha256": source_hash, "images": len(records),
               "classes": dict(sorted(Counter(row["label"] for row in records).items())),
               "physical_groups": len({row["group_id"] for row in records}),
               "exact_duplicate_groups": sum(count > 1 for count in Counter(row["sha256"] for row in records).values()),
               "group_ids_complete": True, "review_provenance_complete": True}
    if not apply:
        return summary
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".intake-{batch_id}-", dir=output.parent) as temporary:
        stage = Path(temporary)
        review_rows = []
        for row in records:
            destination = stage / "raw" / row["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.parent.chmod(0o700)
            shutil.copyfile(source_root / row["source_path"], destination)
            destination.chmod(0o600)
            if digest_file(destination) != row["sha256"]:
                raise ValueError(f"Image changed during intake: {row['source_path']}")
            note = _review_note(row)
            if len(note) > 2000:
                raise ValueError(f"Review evidence too long: {row['source_path']}")
            review_rows.append({"path": row["path"], "current_label": row["label"],
                                "exact_hash_group": row["sha256"], "label_conflict": "",
                                "reviewed_label": row["label"], "group_id": row["group_id"],
                                "reviewed_by": row["reviewed_by"], "reviewed_at": row["reviewed_at"],
                                "review_notes": note})
        review_csv = stage / "review.csv"
        atomic_csv(review_csv, review_rows, FIELDS)
        manifest = {"schema_version": 1, "batch_id": batch_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_csv_sha256": summary["source_csv_sha256"],
                    "review_csv_sha256": digest_file(review_csv), "summary": summary,
                    "images": records}
        canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
        manifest_path = stage / "intake-manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest_path.chmod(0o600)
        (stage / "raw").chmod(0o700)
        verify_intake(stage)
        if digest_file(Path(source_csv)) != source_hash:
            raise ValueError("Capture CSV changed during intake")
        if output.exists():
            raise FileExistsError(f"Intake output appeared during staging: {output}")
        stage.rename(output)
    return {**summary, "output": str(output), "manifest_sha256": manifest["manifest_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--capture-csv", type=Path)
    selection.add_argument("--export-template", type=Path)
    selection.add_argument("--verify-intake", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-id")
    parser.add_argument("--labels-manifest", type=Path, default=ROOT / "models/model-manifest.json")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_intake:
            if args.source_root or args.apply or args.output or args.batch_id:
                parser.error("--verify-intake does not accept capture/export options")
            print(json.dumps(verify_intake(args.verify_intake), ensure_ascii=False, indent=2))
        elif args.export_template:
            if not args.source_root:
                parser.error("--export-template requires --source-root")
            if args.apply or args.output or args.batch_id:
                parser.error("Template export does not accept --apply, --output or --batch-id")
            print(f"Exported {export_template(args.source_root, args.export_template)} images to {args.export_template}")
        else:
            if not args.source_root or not args.output or not args.batch_id:
                parser.error("--capture-csv requires --source-root, --output and --batch-id")
            result = intake(args.source_root, args.capture_csv, args.output, args.batch_id,
                            args.labels_manifest, apply=args.apply)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError, OSError, KeyError) as error:
        parser.exit(1, f"Capture intake rejected: {error}\n")


if __name__ == "__main__":
    main()
