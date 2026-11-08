"""Record dataset contents and detect exact cross-split leakage."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_label(label):
    return label.replace("USA_2 Doolar", "USA_2 Dollar")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/dataset-audit.json"))
    args = parser.parse_args()
    counts = defaultdict(Counter)
    seen = defaultdict(list)
    actual_paths = set()
    manifest_hash = hashlib.sha256()
    for split in ("train", "val", "test"):
        for path in sorted((args.root / split).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            actual_paths.add(str(path.relative_to(args.root)))
            label = canonical_label(path.parent.name)
            image_hash = hash_file(path)
            counts[split][label] += 1
            seen[image_hash].append({"split": split, "label": label, "path": str(path)})
            manifest_hash.update(f"{split}/{label}/{path.name}:{image_hash}\n".encode())
    leaks = [locations for locations in seen.values() if len({item["split"] for item in locations}) > 1]
    conflicting_labels = [locations for locations in seen.values() if len({item["label"] for item in locations}) > 1]
    classes = {split: sorted(counts[split]) for split in counts}
    split_manifest = args.root / "split-manifest.json"
    group_split_verified = False
    manifest_integrity = False
    if split_manifest.exists():
        declared = json.loads(split_manifest.read_text(encoding="utf-8"))
        assignments = declared["assignments"]
        group_splits = defaultdict(set)
        group_labels = defaultdict(set)
        source_paths = set()
        expected_paths = set()
        structurally_valid = declared.get("group_type") == "physical_banknote_or_capture_session"
        for item in assignments:
            group_splits[item["group_id"]].add(item["split"])
            group_labels[item["group_id"]].add(item["label"])
            if item["split"] not in {"train", "val", "test"} or not item["files"]:
                structurally_valid = False
            for source_text in item["files"]:
                source = Path(source_text)
                if source.is_absolute() or ".." in source.parts or source_text in source_paths:
                    structurally_valid = False
                source_paths.add(source_text)
                filename = hashlib.sha256(source_text.encode()).hexdigest()[:12] + source.suffix.lower()
                expected_paths.add(str(Path(item["split"]) / item["label"] / filename))
        manifest_integrity = structurally_valid and len(expected_paths) == len(source_paths) and expected_paths == actual_paths
        group_split_verified = (
            manifest_integrity and bool(group_splits)
            and all(len(splits) == 1 for splits in group_splits.values())
            and all(len(labels) == 1 for labels in group_labels.values())
        )
    output = {
        "dataset_version": manifest_hash.hexdigest(),
        "counts": {split: dict(sorted(counts[split].items())) for split in counts},
        "class_sets_match": len({tuple(values) for values in classes.values()}) == 1,
        "exact_cross_split_duplicates": len(leaks),
        "duplicate_groups": leaks,
        "conflicting_label_groups": len(conflicting_labels),
        "conflicting_label_examples": conflicting_labels[:20],
        "group_split_verified": group_split_verified,
        "split_manifest_integrity": manifest_integrity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dataset {output['dataset_version'][:12]}: {sum(map(sum, (c.values() for c in counts.values())))} images; {len(leaks)} exact cross-split duplicates")
    if not output["class_sets_match"] or leaks or conflicting_labels or not group_split_verified:
        raise SystemExit("Dataset audit failed; inspect output JSON")


if __name__ == "__main__":
    main()
