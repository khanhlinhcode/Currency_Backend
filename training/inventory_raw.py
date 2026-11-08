"""Inventory raw images and prepare a human review queue for group IDs and labels."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--output", type=Path, default=Path("results/raw-inventory.json"))
    parser.add_argument("--review-csv", type=Path, default=Path("results/group-review.csv"))
    args = parser.parse_args()
    if not args.raw.is_dir():
        parser.error(f"Raw dataset directory does not exist: {args.raw}")

    records = []
    by_hash = defaultdict(list)
    class_counts = Counter()
    for image in sorted(args.raw.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in EXTENSIONS:
            continue
        relative = image.relative_to(args.raw)
        if len(relative.parts) != 3:
            raise SystemExit(f"Expected country/denomination/image: {relative}")
        label = f"{relative.parts[0]}_{relative.parts[1]}".replace("USA_2 Doolar", "USA_2 Dollar")
        digest = sha256(image)
        record = {"path": str(relative), "label": label, "sha256": digest}
        records.append(record)
        by_hash[digest].append(record)
        class_counts[label] += 1

    exact_duplicate_groups = [group for group in by_hash.values() if len(group) > 1]
    conflicting_groups = [group for group in exact_duplicate_groups if len({item["label"] for item in group}) > 1]
    conflicts = {item["path"] for group in conflicting_groups for item in group}
    summary = {
        "raw_root": str(args.raw),
        "images": len(records),
        "classes": dict(sorted(class_counts.items())),
        "unique_content_hashes": len(by_hash),
        "exact_duplicate_groups": len(exact_duplicate_groups),
        "duplicate_images": sum(len(group) for group in exact_duplicate_groups),
        "conflicting_label_groups": len(conflicting_groups),
        "conflicting_label_images": len(conflicts),
        "conflicting_groups": conflicting_groups,
        "note": "Exact content hashes do not establish physical-banknote or capture-session identity",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.review_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.review_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["path", "current_label", "exact_hash_group", "label_conflict", "reviewed_label", "group_id", "review_notes"])
        writer.writeheader()
        for record in records:
            writer.writerow({
                "path": record["path"],
                "current_label": record["label"],
                "exact_hash_group": record["sha256"],
                "label_conflict": "yes" if record["path"] in conflicts else "",
                "reviewed_label": "",
                "group_id": "",
                "review_notes": "",
            })
    print(f"Raw inventory: {len(records)} images, {len(exact_duplicate_groups)} exact-duplicate groups, {len(conflicting_groups)} conflicting-label groups")
    print(f"Review files: {args.output}, {args.review_csv}")


if __name__ == "__main__":
    main()
