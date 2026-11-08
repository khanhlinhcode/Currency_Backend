"""Summarize recent model behavior without treating traffic mix as proven drift."""

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from api.config import DATABASE_URL
from api.models import PredictionHistory


def jensen_shannon(left, right):
    keys = set(left) | set(right)
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total or not right_total:
        return None
    p = {key: left.get(key, 0) / left_total for key in keys}
    q = {key: right.get(key, 0) / right_total for key in keys}
    middle = {key: (p[key] + q[key]) / 2 for key in keys}

    def divergence(distribution):
        return sum(value * math.log2(value / middle[key]) for key, value in distribution.items() if value)

    return round((divergence(p) + divergence(q)) / 2, 6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("models/model-manifest.json"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--max-low-confidence-increase", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=Path("results/drift-report.json"))
    args = parser.parse_args()
    if args.window < 1 or args.min_samples < 1 or args.max_low_confidence_increase < 0:
        parser.error("Window and minimum samples must be positive; increase cannot be negative")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if evaluation["model_version"] != manifest["version"]:
        raise SystemExit("Evaluation model version does not match manifest")
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        rows = session.execute(
            select(PredictionHistory)
            .where(PredictionHistory.model_version == manifest["version"], PredictionHistory.requested_by.notlike("benchmark:%"))
            .order_by(PredictionHistory.id.desc())
            .limit(args.window)
        ).scalars().all()
    engine.dispose()

    predictions = Counter(row.prediction for row in rows)
    low_confidence = sum(row.confidence < manifest["confidence_threshold"] for row in rows)
    reviewed = [row for row in rows if row.actual_label is not None]
    reference = audit["counts"]["test"]
    baseline_low_confidence = 1 - evaluation["accepted_coverage"]
    observed_low_confidence = low_confidence / len(rows) if rows else None
    signal = (
        "insufficient_samples" if len(rows) < args.min_samples else
        "investigate_low_confidence" if observed_low_confidence > baseline_low_confidence + args.max_low_confidence_increase else
        "within_threshold"
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": manifest["version"],
        "release_status": manifest["release_status"],
        "window": args.window,
        "sample_size": len(rows),
        "baseline_test_low_confidence_rate": round(baseline_low_confidence, 6),
        "observed_low_confidence_rate": round(observed_low_confidence, 6) if observed_low_confidence is not None else None,
        "allowed_absolute_increase": args.max_low_confidence_increase,
        "prediction_distribution": dict(sorted(predictions.items())),
        "test_label_distribution": reference,
        "distribution_js_divergence": jensen_shannon(predictions, reference),
        "reviewed_count": len(reviewed),
        "reviewed_accuracy": round(sum(row.actual_label == row.prediction for row in reviewed) / len(reviewed), 6) if reviewed else None,
        "signal": signal,
        "interpretation": "Class-mix shift and review accuracy are diagnostic; traffic and reviewed samples can differ from the held-out test set",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model_version", "sample_size", "observed_low_confidence_rate", "reviewed_count", "reviewed_accuracy", "distribution_js_divergence", "signal")}, indent=2))
    if signal == "investigate_low_confidence":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
