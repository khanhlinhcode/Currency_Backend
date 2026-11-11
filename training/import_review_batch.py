"""Import provenance-backed group decisions into the local human review CSV.

No group is inferred from a filename, image hash, EXIF, or class label. The
input is a reviewer-attested mapping from source records. Check mode is the
default; --apply performs one atomic CSV update after the same validation.
"""

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from training.review_app import FIELDS, digest_file
from training.review_lock import review_file_lock
from training.review_queue import generate_review_queue


IMPORT_FIELDS = ["path", "sha256", "current_label", "reviewed_label", "group_id",
                 "source_ref", "reviewed_by", "reviewed_at", "review_notes"]


def read_csv(path, fields):
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != fields:
            raise ValueError(f"CSV columns must be: {', '.join(fields)}")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError("CSV has missing or extra cells")
        return rows


def atomic_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".review-batch-",
                                         suffix=".csv", delete=False, newline="",
                                         encoding="utf-8") as target:
            temporary = Path(target.name)
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            target.flush()
            os.fsync(target.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_template(review_csv, output):
    rows = read_csv(review_csv, FIELDS)
    template = [{"path": row["path"], "sha256": row["exact_hash_group"],
                 "current_label": row["current_label"],
                 "reviewed_label": row["reviewed_label"], "group_id": "",
                 "source_ref": "", "reviewed_by": "", "reviewed_at": "",
                 "review_notes": ""} for row in rows if not row["group_id"]]
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Template already exists: {output}")
    atomic_csv(output, template, IMPORT_FIELDS)
    return len(template)


def clean(value, limit, field):
    value = value.strip()
    if len(value) > limit or any(ord(char) < 32 for char in value):
        raise ValueError(f"Invalid {field}")
    return value


def normalized_time(value):
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("reviewed_at must be ISO-8601 with timezone") from error
    if parsed.tzinfo is None:
        raise ValueError("reviewed_at must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def prepare(review_rows, source_rows, raw_root):
    indexed = {row["path"]: row for row in review_rows}
    if len(indexed) != len(review_rows):
        raise ValueError("Duplicate paths in review CSV")
    labels = {row["current_label"] for row in review_rows}
    proposed = [dict(row) for row in review_rows]
    by_path = {row["path"]: row for row in proposed}
    seen = set()
    ready = 0
    changed = 0
    for source in source_rows:
        path = source["path"]
        if path in seen:
            raise ValueError(f"Duplicate import path: {path}")
        seen.add(path)
        current = indexed.get(path)
        if current is None or source["sha256"] != current["exact_hash_group"] or source["current_label"] != current["current_label"]:
            raise ValueError(f"Unknown or stale image record: {path}")
        group = clean(source["group_id"], 128, "group_id")
        if not group:
            continue  # Blank template rows are deliberately not decisions.
        ready += 1
        relative = Path(path)
        if relative.is_absolute() or len(relative.parts) != 3 or ".." in relative.parts:
            raise ValueError(f"Unsafe image path: {path}")
        image = (Path(raw_root) / relative).resolve(strict=True)
        if not image.is_relative_to(Path(raw_root).resolve(strict=True)) or digest_file(image) != source["sha256"]:
            raise ValueError(f"Raw image changed: {path}")
        reference = clean(source["source_ref"], 300, "source_ref")
        reviewer = clean(source["reviewed_by"], 80, "reviewed_by")
        if not reference or len(reviewer) < 2:
            raise ValueError(f"Source reference and reviewer are required: {path}")
        reviewed_at = normalized_time(source["reviewed_at"])
        label = clean(source["reviewed_label"], 100, "reviewed_label")
        if label and label not in labels:
            raise ValueError(f"Unknown reviewed label: {path}")
        old = by_path[path]
        if old["group_id"] and old["group_id"] != group:
            raise ValueError(f"Existing group differs; review manually: {path}")
        if old["reviewed_label"] and label and old["reviewed_label"] != label:
            raise ValueError(f"Existing reviewed label differs; review manually: {path}")
        if old["group_id"] == group and (not label or old["reviewed_label"] == label):
            continue  # Idempotent replay; never overwrite the original attestation.
        note = clean(source["review_notes"], 1000, "review_notes")
        attestation = f"Group source: {reference}; attested by {reviewer} at {reviewed_at}"
        combined = "\n".join(part for part in (old["review_notes"], note, attestation) if part)
        if len(combined) > 2000:
            raise ValueError(f"Review notes exceed 2,000 characters: {path}")
        old.update(group_id=group, reviewed_label=label or old["reviewed_label"],
                   reviewed_by=reviewer, reviewed_at=reviewed_at, review_notes=combined)
        changed += 1
    if ready == 0:
        raise ValueError("No group_id decisions found in input CSV")
    by_hash = defaultdict(list)
    for row in proposed:
        by_hash[row["exact_hash_group"]].append(row)
    for digest, copies in by_hash.items():
        groups = {row["group_id"] for row in copies if row["group_id"]}
        labels_for_copies = {row["reviewed_label"] or row["current_label"] for row in copies}
        if len(groups) > 1 or len(labels_for_copies) > 1:
            raise ValueError(f"Exact image copies disagree on group or label: {digest}")
    return proposed, ready, changed


def import_batch(raw_root, review_csv, input_csv, queue_csv, status_json, report_json, apply=False):
    review_csv = Path(review_csv)
    input_csv = Path(input_csv)
    source_rows = read_csv(input_csv, IMPORT_FIELDS)
    with review_file_lock(review_csv):
        review_rows = read_csv(review_csv, FIELDS)
        before = digest_file(review_csv)
        proposed, ready, changed = prepare(review_rows, source_rows, raw_root)
        result = {"schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                  "mode": "apply" if apply else "check", "input_sha256": digest_file(input_csv),
                  "review_csv_before_sha256": before, "import_rows": len(source_rows),
                  "attested_rows": ready, "new_decisions": changed,
                  "remaining_group_ids": sum(not row["group_id"] for row in proposed),
                  "pending_label_conflicts": sum(row["label_conflict"] == "yes" and not row["reviewed_label"] for row in proposed)}
        if apply and changed:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = review_csv.with_name(f"{review_csv.stem}.before-batch-{stamp}-{before[:12]}.csv")
            if backup.exists():
                raise FileExistsError(f"Backup already exists: {backup}")
            shutil.copy2(review_csv, backup)
            backup.chmod(0o600)
            if digest_file(review_csv) != before:
                raise ValueError("Review CSV changed during import; retry")
            atomic_csv(review_csv, proposed, FIELDS)
            result["backup_path"] = str(backup)
        if apply:
            generate_review_queue(review_csv, Path(queue_csv), Path(status_json))
        result["review_csv_after_sha256"] = digest_file(review_csv) if apply else before
    report_json = Path(report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_json.chmod(0o600)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--export-template", type=Path)
    selection.add_argument("--input", type=Path)
    parser.add_argument("--apply", action="store_true", help="Commit validated decisions; default is check only")
    parser.add_argument("--raw", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--review-csv", type=Path, default=Path("results/group-review.csv"))
    parser.add_argument("--queue", type=Path, default=Path("results/review-queue.csv"))
    parser.add_argument("--status", type=Path, default=Path("results/review-status.json"))
    parser.add_argument("--report", type=Path, default=Path("results/review-batch-report.json"))
    args = parser.parse_args()
    try:
        if args.export_template:
            if args.apply:
                parser.error("--apply cannot be used with --export-template")
            print(f"Exported {export_template(args.review_csv, args.export_template)} rows to {args.export_template}")
        else:
            print(json.dumps(import_batch(args.raw, args.review_csv, args.input,
                                          args.queue, args.status, args.report, args.apply),
                             ensure_ascii=False))
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as error:
        parser.exit(1, f"Review batch rejected: {error}\n")


if __name__ == "__main__":
    main()
