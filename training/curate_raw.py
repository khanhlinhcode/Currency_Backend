"""Copy an exact-deduplicated raw snapshot for human grouping."""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from review_queue import review_issues


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
    parser.add_argument("--review-csv", type=Path, default=Path("results/group-review.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/curated_raw"))
    parser.add_argument("--group-template", type=Path, default=Path("results/curated-groups.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/curation-report.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}; choose a new snapshot path")

    with args.review_csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    by_hash = defaultdict(list)
    known_labels = {row["current_label"] for row in rows}
    source_paths = set()
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 3:
            raise SystemExit(f"Invalid source path: {relative}")
        if row["path"] in source_paths or not (args.raw / relative).is_file():
            raise SystemExit(f"Duplicate or missing source path: {relative}")
        if sha256(args.raw / relative) != row["exact_hash_group"]:
            raise SystemExit(f"Raw image changed since inventory: {relative}")
        source_paths.add(row["path"])
        if row["reviewed_label"] and row["reviewed_label"] not in known_labels:
            raise SystemExit(f"Unknown reviewed label: {row['reviewed_label']}")
        by_hash[row["exact_hash_group"]].append(row)
    raw_paths = {str(path.relative_to(args.raw)) for path in args.raw.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS}
    if source_paths != raw_paths:
        raise SystemExit(f"Review CSV does not cover raw dataset: {len(raw_paths - source_paths)} missing, {len(source_paths - raw_paths)} extra")

    included = []
    quarantined = []
    for digest, group in sorted(by_hash.items()):
        chosen_labels = {row["reviewed_label"] or row["current_label"] for row in group}
        has_conflict = any(row["label_conflict"] == "yes" for row in group)
        fully_reviewed = all(row["reviewed_label"] for row in group)
        if has_conflict and not fully_reviewed:
            quarantined.append({"sha256": digest, "paths": [row["path"] for row in group], "reason": "label conflict awaiting review"})
            continue
        if len(chosen_labels) != 1:
            raise SystemExit(f"Reviewed labels disagree within exact duplicate group: {digest}")
        representative = sorted(group, key=lambda row: row["path"])[0]
        group_ids = {row.get("group_id", "").strip() for row in group}
        if len(group_ids) > 1 and any(group_ids):
            raise SystemExit(f"Exact copies have conflicting group IDs: {digest}")
        label = chosen_labels.pop()
        country, denomination = label.split("_", 1)
        destination = Path(country) / denomination / Path(representative["path"]).name
        included.append({
            "source": representative["path"], "path": str(destination), "label": label,
            "sha256": digest, "source_copies": len(group),
            "group_id": representative.get("group_id", "").strip(),
            "reviewed_by": representative.get("reviewed_by", "").strip(),
            "reviewed_at": representative.get("reviewed_at", "").strip(),
            "source_paths": json.dumps(sorted(row["path"] for row in group)),
            "review_complete": not any(review_issues(row) for row in group),
        })

    destinations = [item["path"] for item in included]
    if len(set(destinations)) != len(destinations):
        raise SystemExit("Two curated images would have the same output path")
    args.output.mkdir(parents=True, exist_ok=False)
    for item in included:
        target = args.output / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.raw / item["source"], target)

    args.group_template.parent.mkdir(parents=True, exist_ok=True)
    with args.group_template.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["path", "group_id", "label", "sha256", "source", "source_copies", "reviewed_by", "reviewed_at", "source_paths", "review_complete"])
        writer.writeheader()
        writer.writerows(included)
    class_counts = Counter(item["label"] for item in included)
    report = {
        "raw_images": len(rows),
        "curated_images": len(included),
        "exact_duplicate_copies_removed": len(rows) - len(included) - sum(len(group["paths"]) for group in quarantined),
        "quarantined_groups": quarantined,
        "quarantined_images": sum(len(group["paths"]) for group in quarantined),
        "class_counts": dict(sorted(class_counts.items())),
        "group_ids_complete": all(item["group_id"] for item in included),
        "human_review_complete": not quarantined and all(item["review_complete"] for item in included),
        "review_csv_sha256": sha256(args.review_csv),
        "note": "Group IDs must identify a physical banknote or capture session; exact hashes alone are insufficient",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Curated {len(included)} unique images; quarantined {report['quarantined_images']} conflicting-label images; group IDs complete: {report['group_ids_complete']}")


if __name__ == "__main__":
    main()
