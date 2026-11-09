"""Build train/val/test splits without sharing a physical note across splits.

The CSV is supplied by the data labeling process. `group_id` is one physical
banknote (or one capture session when the physical ID is unavailable).
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def canonical_label(path):
    return f"{path.parts[0]}_{path.parts[1]}".replace("USA_2 Doolar", "USA_2 Dollar")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--groups", type=Path, required=True, help="CSV columns: path,group_id")
    parser.add_argument("--output", type=Path, default=Path("dataset/processed_v2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-version", help="Required for a release-grade v2 dataset")
    parser.add_argument("--curation-report", type=Path, help="Curation report proving no unresolved review items")
    parser.add_argument("--attested-physical-groups", action="store_true", help="Confirm group_id was reviewed as a physical note or capture session")
    args = parser.parse_args()

    if args.dataset_version:
        if not args.attested_physical_groups or not args.curation_report:
            parser.error("Versioned dataset requires --attested-physical-groups and --curation-report")
        curation = json.loads(args.curation_report.read_text(encoding="utf-8"))
        if curation.get("quarantined_images") != 0 or not curation.get("human_review_complete"):
            raise SystemExit("Human review incomplete or conflicting images quarantined")
    group_ids = {}
    review_rows = {}
    with args.groups.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 3:
                raise SystemExit(f"Invalid relative path: {relative}")
            if not row["group_id"].strip():
                raise SystemExit(f"Missing group_id: {relative}")
            if relative in group_ids:
                raise SystemExit(f"Duplicate CSV row: {relative}")
            group_ids[relative] = row["group_id"].strip()
            review_rows[relative] = row
            if args.dataset_version:
                try:
                    reviewed_at = datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00"))
                except (KeyError, ValueError):
                    raise SystemExit(f"Invalid reviewed_at: {relative}")
                if (not reviewed_at.tzinfo or not row.get("reviewed_by", "").strip()
                    or row.get("review_complete") != "True" or row.get("label") != canonical_label(relative)
                    or row.get("sha256") != file_sha256(args.raw / relative)):
                    raise SystemExit(f"Missing review evidence or changed image: {relative}")
                try:
                    sources = json.loads(row["source_paths"])
                except (KeyError, ValueError):
                    raise SystemExit(f"Missing source provenance: {relative}")
                if not sources or row.get("source") not in sources or len(sources) != int(row.get("source_copies", 0)):
                    raise SystemExit(f"Invalid source provenance: {relative}")

    images = sorted(path.relative_to(args.raw) for path in args.raw.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if set(images) != set(group_ids):
        missing = set(images) - set(group_ids)
        extra = set(group_ids) - set(images)
        raise SystemExit(f"Group CSV incomplete: {len(missing)} missing, {len(extra)} extra")

    groups = defaultdict(list)
    group_labels = {}
    for relative in images:
        if len(relative.parts) < 3:
            raise SystemExit(f"Expected country/denomination/image path: {relative}")
        label = canonical_label(relative)
        group_id = group_ids[relative]
        if group_id in group_labels and group_labels[group_id] != label:
            raise SystemExit(f"Physical group {group_id} spans different labels")
        group_labels[group_id] = label
        groups[(label, group_id)].append(relative)

    by_label = defaultdict(list)
    for (label, group_id), members in groups.items():
        by_label[label].append((group_id, members))

    rng = random.Random(args.seed)
    assignments = []
    for label, label_groups in sorted(by_label.items()):
        if len(label_groups) < 3:
            raise SystemExit(f"Class {label} needs at least three physical groups")
        rng.shuffle(label_groups)
        n = len(label_groups)
        train_end = min(n - 2, max(1, round(n * 0.70)))
        val_end = min(n - 1, train_end + max(1, round(n * 0.15)))
        for index, (group_id, members) in enumerate(label_groups):
            split = "train" if index < train_end else "val" if index < val_end else "test"
            assignments.append({"split": split, "label": label, "group_id": group_id, "files": [str(path) for path in members]})

    args.output.mkdir(parents=True, exist_ok=False)
    image_entries = []
    for item in assignments:
        destination = args.output / item["split"] / item["label"]
        destination.mkdir(parents=True, exist_ok=True)
        for relative_text in item["files"]:
            relative = Path(relative_text)
            unique_name = hashlib.sha256(str(relative).encode()).hexdigest()[:12] + relative.suffix.lower()
            target = destination / unique_name
            shutil.copy2(args.raw / relative, target)
            if args.dataset_version:
                row = review_rows[relative]
                image_entries.append({"path": str(target.relative_to(args.output)), "source": str(relative),
                    "source_paths": json.loads(row["source_paths"]), "sha256": row["sha256"],
                    "label": item["label"], "group_id": item["group_id"], "split": item["split"],
                    "reviewed_by": row["reviewed_by"].strip(), "reviewed_at": row["reviewed_at"]})
    group_type = "physical_banknote_or_capture_session" if args.attested_physical_groups else "unverified_group_ids"
    manifest = {"seed": args.seed, "group_type": group_type, "group_csv_sha256": hashlib.sha256(args.groups.read_bytes()).hexdigest(), "assignments": assignments}
    split_path = args.output / "split-manifest.json"
    split_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dataset_version:
        from collections import Counter
        counts = {split: dict(sorted(Counter(item["label"] for item in image_entries if item["split"] == split).items()))
                  for split in ("train", "val", "test")}
        dataset_manifest = {"schema_version": 2, "version": args.dataset_version,
            "created_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
            "group_type": group_type, "counts": counts,
            "curation_report_sha256": file_sha256(args.curation_report),
            "group_csv_sha256": file_sha256(args.groups), "split_manifest_sha256": file_sha256(split_path),
            "images": sorted(image_entries, key=lambda entry: entry["path"])}
        canonical = json.dumps(dataset_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        dataset_manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
        (args.output / "dataset-manifest.json").write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Created {len(assignments)} disjoint groups in {args.output} ({group_type})")


if __name__ == "__main__":
    main()
