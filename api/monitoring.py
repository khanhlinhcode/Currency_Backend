"""Apply explicit staging monitoring thresholds without inventing a drift baseline."""

import math
from collections import Counter


def jensen_shannon(left, right):
    keys = set(left) | set(right)
    left_total, right_total = sum(left.values()), sum(right.values())
    if not left_total or not right_total:
        return None
    p = {key: left.get(key, 0) / left_total for key in keys}
    q = {key: right.get(key, 0) / right_total for key in keys}
    middle = {key: (p[key] + q[key]) / 2 for key in keys}
    def divergence(distribution):
        return sum(value * math.log2(value / middle[key]) for key, value in distribution.items() if value)
    return round((divergence(p) + divergence(q)) / 2, 6)


def assess(stats, predictions, baseline, errors_24h, storage_bytes, targets):
    reasons = []
    processing_p95 = stats.get("processing_latency", {}).get("p95_ms")
    processing_p99 = stats.get("processing_latency", {}).get("p99_ms")
    status = "healthy"
    if errors_24h >= targets["critical_api_5xx_24h"] or (
        processing_p95 is not None and processing_p95 >= targets["critical_processing_p95_ms"]
    ):
        status = "critical"
        reasons.append("API errors or processing p95 exceeded critical threshold")
    elif errors_24h >= targets["warning_api_5xx_24h"] or (
        processing_p95 is not None and processing_p95 >= targets["warning_processing_p95_ms"]
    ):
        status = "warning"
        reasons.append("API errors or processing p95 exceeded warning threshold")
    if storage_bytes is None:
        if status == "healthy":
            status = "warning"
        reasons.append("Private storage usage unavailable")
    elif storage_bytes >= targets["warning_storage_bytes"]:
        if status == "healthy":
            status = "warning"
        reasons.append("Private storage usage exceeded warning threshold")

    count = len(predictions)
    low_confidence_rate = (sum(item["confidence"] < item["threshold"] for item in predictions) / count) if count else None
    class_mix = dict(Counter(item["prediction"] for item in predictions))
    drift_js = None
    drift_status = "insufficient_data"
    if baseline and count >= targets["minimum_model_samples"]:
        drift_js = jensen_shannon(class_mix, baseline["class_counts"])
        baseline_low = 1 - baseline["accepted_coverage"]
        if low_confidence_rate > baseline_low + targets["max_low_confidence_increase"] or (
            drift_js is not None and drift_js > targets["warning_distribution_js"]
        ):
            drift_status = "warning"
            if status == "healthy":
                status = "warning"
            reasons.append("Model confidence or class mix requires investigation")
        else:
            drift_status = "healthy"
    else:
        if status == "healthy":
            status = "insufficient_data"
        reasons.append("No verified model baseline or fewer than minimum operational samples")
    reviewed_count = stats.get("reviewed_count", 0)
    return {
        "status": status, "reasons": reasons, "drift_status": drift_status,
        "baseline_available": baseline is not None, "operational_sample_size": count,
        "reviewed_count": reviewed_count,
        "reviewed_accuracy": stats.get("reviewed_accuracy") if reviewed_count >= targets["minimum_reviewed_samples"] else None,
        "reviewed_accuracy_status": "available" if reviewed_count >= targets["minimum_reviewed_samples"] else "insufficient_data",
        "low_confidence_rate": round(low_confidence_rate, 6) if low_confidence_rate is not None else None,
        "class_distribution": class_mix, "distribution_js_divergence": drift_js,
        "processing_p95_ms": processing_p95, "processing_p99_ms": processing_p99,
        "inference_p95_ms": stats.get("inference_latency", {}).get("p95_ms"),
        "inference_p99_ms": stats.get("inference_latency", {}).get("p99_ms"),
        "api_5xx_24h": errors_24h, "storage_bytes": storage_bytes,
    }
