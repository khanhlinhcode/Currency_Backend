"""Produce a deterministic human review queue; never infer physical group IDs."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def review_issues(row):
    issues = []
    if row.get("label_conflict") == "yes" and not row.get("reviewed_label", "").strip():
        issues.append("label_conflict")
    if not row.get("group_id", "").strip():
        issues.append("group_id")
    if not row.get("reviewed_by", "").strip():
        issues.append("reviewed_by")
    reviewed_at = row.get("reviewed_at", "").strip()
    try:
        if not reviewed_at or datetime.fromisoformat(reviewed_at.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError
    except ValueError:
        issues.append("reviewed_at_utc")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, default=Path("results/group-review.csv"))
    parser.add_argument("--queue", type=Path, default=Path("results/review-queue.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/review-status.json"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    with args.review_csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len({row["path"] for row in rows}) != len(rows):
        raise SystemExit("Duplicate review paths")
    pending = []
    for row in rows:
        issues = review_issues(row)
        if issues:
            pending.append({"path": row["path"], "current_label": row["current_label"],
                            "sha256": row["exact_hash_group"], "issues": ",".join(issues),
                            "reviewed_label": row.get("reviewed_label", ""),
                            "group_id": row.get("group_id", ""), "review_notes": row.get("review_notes", "")})
    args.queue.parent.mkdir(parents=True, exist_ok=True)
    with args.queue.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["path", "current_label", "sha256", "issues", "reviewed_label", "group_id", "review_notes"])
        writer.writeheader()
        writer.writerows(sorted(pending, key=lambda item: item["path"]))
    report = {"review_rows": len(rows), "complete_rows": len(rows) - len(pending),
              "pending_rows": len(pending), "pending_label_conflicts": sum("label_conflict" in row["issues"] for row in pending),
              "pending_group_ids": sum("group_id" in row["issues"] for row in pending),
              "queue": str(args.queue), "note": "Only a human with source evidence can attest physical note or capture session groups"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if args.require_complete and pending:
        raise SystemExit("Human review is incomplete")


if __name__ == "__main__":
    main()
