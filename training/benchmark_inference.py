"""Measure warm inference latency using the exact API preprocessing path."""

import argparse
import json
import math
import platform
import time
from pathlib import Path

from PIL import Image

from api.inference import MODEL_VERSION, preprocess_image
from api.model_loader import load_model


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")
    tensor = preprocess_image(image)
    model = load_model()
    for _ in range(args.warmup):
        model.predict(tensor, verbose=0)
    timings = []
    for _ in range(args.runs):
        started = time.perf_counter()
        model.predict(tensor, verbose=0)
        timings.append((time.perf_counter() - started) * 1000)
    result = {
        "model_version": MODEL_VERSION,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "runs": args.runs,
        "mean_ms": round(sum(timings) / len(timings), 2),
        "p50_ms": round(percentile(timings, 0.50), 2),
        "p95_ms": round(percentile(timings, 0.95), 2),
        "p99_ms": round(percentile(timings, 0.99), 2),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
