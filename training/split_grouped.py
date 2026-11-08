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
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def canonical_label(path):
    return f"{path.parts[0]}_{path.parts[1]}".replace("USA_2 Doolar", "USA_2 Dollar")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--groups", type=Path, required=True, help="CSV columns: path,group_id")
    parser.add_argument("--output", type=Path, default=Path("dataset/processed_v2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attested-physical-groups", action="store_true", help="Confirm group_id was reviewed as a physical note or capture session")
    args = parser.parse_args()

    group_ids = {}
    with args.groups.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise SystemExit(f"Invalid relative path: {relative}")
            if not row["group_id"].strip():
                raise SystemExit(f"Missing group_id: {relative}")
            if relative in group_ids:
                raise SystemExit(f"Duplicate CSV row: {relative}")
            group_ids[relative] = row["group_id"].strip()

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
    for item in assignments:
        destination = args.output / item["split"] / item["label"]
        destination.mkdir(parents=True, exist_ok=True)
        for relative_text in item["files"]:
            relative = Path(relative_text)
            unique_name = hashlib.sha256(str(relative).encode()).hexdigest()[:12] + relative.suffix.lower()
            shutil.copy2(args.raw / relative, destination / unique_name)
    group_type = "physical_banknote_or_capture_session" if args.attested_physical_groups else "unverified_group_ids"
    manifest = {"seed": args.seed, "group_type": group_type, "group_csv_sha256": hashlib.sha256(args.groups.read_bytes()).hexdigest(), "assignments": assignments}
    (args.output / "split-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created {len(assignments)} disjoint groups in {args.output} ({group_type})")


if __name__ == "__main__":
    main()
