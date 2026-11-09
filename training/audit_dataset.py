"""Record dataset contents and detect exact cross-split leakage."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
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
    actual_hashes = {}
    manifest_hash = hashlib.sha256()
    for split in ("train", "val", "test"):
        for path in sorted((args.root / split).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            actual_paths.add(str(path.relative_to(args.root)))
            label = canonical_label(path.parent.name)
            image_hash = hash_file(path)
            actual_hashes[str(path.relative_to(args.root))] = image_hash
            counts[split][label] += 1
            seen[image_hash].append({"split": split, "label": label, "path": str(path)})
            manifest_hash.update(f"{split}/{label}/{path.name}:{image_hash}\n".encode())
    leaks = [locations for locations in seen.values() if len({item["split"] for item in locations}) > 1]
    conflicting_labels = [locations for locations in seen.values() if len({item["label"] for item in locations}) > 1]
    classes = {split: sorted(counts[split]) for split in counts}
    split_manifest = args.root / "split-manifest.json"
    group_split_verified = False
    manifest_integrity = False
    declared = None
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
    dataset_manifest_path = args.root / "dataset-manifest.json"
    dataset_manifest_verified = False
    human_review_complete = False
    dataset_version = manifest_hash.hexdigest()
    dataset_manifest_sha256 = None
    if dataset_manifest_path.exists() and declared is not None:
        try:
            dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
            declared_digest = dataset_manifest.pop("manifest_sha256")
            canonical = json.dumps(dataset_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            rows = dataset_manifest["images"]
            paths = [item["path"] for item in rows]
            assignments_by_source = {
                source: (item["split"], item["label"], item["group_id"])
                for item in declared["assignments"] for source in item["files"]
            }
            human_review_complete = all(
                row.get("group_id") and row.get("reviewed_by") and row.get("source")
                and row.get("source_paths") and row["source"] in row["source_paths"]
                and datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00")).tzinfo is not None
                for row in rows
            )
            dataset_manifest_verified = (
                dataset_manifest.get("schema_version") == 2
                and isinstance(dataset_manifest.get("version"), str) and bool(dataset_manifest["version"].strip())
                and declared_digest == hashlib.sha256(canonical).hexdigest()
                and dataset_manifest.get("group_type") == "physical_banknote_or_capture_session"
                and dataset_manifest.get("split_manifest_sha256") == hash_file(split_manifest)
                and dataset_manifest.get("group_csv_sha256") == declared.get("group_csv_sha256")
                and len(paths) == len(set(paths)) and set(paths) == actual_paths
                and len({row["source"] for row in rows}) == len(rows)
                and {row["source"] for row in rows} == set(assignments_by_source)
                and all(row["sha256"] == actual_hashes[row["path"]]
                        and (row["split"], row["label"], row["group_id"]) == assignments_by_source[row["source"]]
                        and row["path"] == str(Path(row["split"]) / row["label"] /
                            (hashlib.sha256(row["source"].encode()).hexdigest()[:12] + Path(row["source"]).suffix.lower()))
                        for row in rows)
                and dataset_manifest.get("counts") == {split: dict(sorted(counts[split].items())) for split in ("train", "val", "test")}
                and human_review_complete and group_split_verified
            )
            if dataset_manifest_verified:
                dataset_version = dataset_manifest["version"]
                dataset_manifest_sha256 = declared_digest
        except (KeyError, ValueError, TypeError, AttributeError):
            dataset_manifest_verified = False
    output = {
        "dataset_version": dataset_version,
        "dataset_content_sha256": manifest_hash.hexdigest(),
        "counts": {split: dict(sorted(counts[split].items())) for split in counts},
        "class_sets_match": len({tuple(values) for values in classes.values()}) == 1,
        "exact_cross_split_duplicates": len(leaks),
        "duplicate_groups": leaks,
        "conflicting_label_groups": len(conflicting_labels),
        "conflicting_label_examples": conflicting_labels[:20],
        "group_split_verified": group_split_verified,
        "split_manifest_integrity": manifest_integrity,
        "dataset_manifest_verified": dataset_manifest_verified,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "human_review_complete": human_review_complete,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dataset {output['dataset_version'][:12]}: {sum(map(sum, (c.values() for c in counts.values())))} images; {len(leaks)} exact cross-split duplicates")
    if not output["class_sets_match"] or leaks or conflicting_labels or not group_split_verified or (dataset_manifest_path.exists() and not dataset_manifest_verified):
        raise SystemExit("Dataset audit failed; inspect output JSON")


if __name__ == "__main__":
    main()
