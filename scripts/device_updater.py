"""Poll an approved release channel and atomically install a verified model package.

Configure CURRENCY_DEVICE_TOKEN in the device's secret store. The serving process
reads current.json and reloads when its generation changes. A failed download or
self-test leaves the previous pointer in place.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path


def request(url, token, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    method = "POST" if payload is not None else "GET"
    req = urllib.request.Request(url, data=data, method=method,
        headers={"X-Device-Token": token, "Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=120)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".pointer-", delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def sync_once(base, device_id, token, state_dir, self_test):
    manifest_url = f"{base}/devices/{device_id}/manifest"
    with request(manifest_url, token) as response:
        release = json.load(response)
    if release.get("status") == "no_release":
        return "no_release"
    run_id = release["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9-]{1,100}", run_id):
        raise ValueError("Invalid release run ID")
    suffix = Path(release["model_filename"]).suffix
    if suffix not in {".onnx", ".keras"}:
        raise ValueError("Unsupported model file type")
    expected = release["model_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Invalid model digest")
    pointer_path = state_dir / "current.json"
    current = json.loads(pointer_path.read_text()) if pointer_path.exists() else {}
    current_model = Path(current.get("model_path", "")) if current.get("model_path") else None
    if (current.get("generation") == release["generation"] and current.get("model_sha256") == expected
        and current_model and current_model.is_file() and sha256(current_model) == expected):
        heartbeat(base, device_id, token, "ready", run_id, release["generation"])
        return "up_to_date"
    destination = state_dir / "releases" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / ("model" + suffix)
    if not model_path.exists() or sha256(model_path) != expected:
        with tempfile.NamedTemporaryFile(dir=destination, prefix=".download-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with request(base + release["artifact_path"], token) as source:
                    digest = hashlib.sha256()
                    while chunk := source.read(1024 * 1024):
                        stream.write(chunk)
                        digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
                if digest.hexdigest() != expected:
                    raise ValueError("Model checksum mismatch")
                os.replace(temporary, model_path)
            finally:
                temporary.unlink(missing_ok=True)
    if self_test:
        result = subprocess.run([*self_test, str(model_path)], capture_output=True, text=True,
                                timeout=120, check=False)
        if result.returncode:
            raise RuntimeError("Device model self-test failed: " + result.stderr[-300:])
    atomic_json(destination / "manifest.json", release["manifest"])
    atomic_json(pointer_path, {"generation": release["generation"], "run_id": run_id,
        "model_sha256": expected, "model_path": str(model_path),
        "manifest_path": str(destination / "manifest.json"),
        "previous_run_id": current.get("run_id")})
    heartbeat(base, device_id, token, "ready", run_id, release["generation"])
    return "updated"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def heartbeat(base, device_id, token, status, run_id=None, generation=None):
    with request(f"{base}/devices/{device_id}/heartbeat", token,
                 {"status": status, "run_id": run_id, "generation": generation}) as response:
        response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--self-test", nargs="+", help="Command and arguments; model path appended")
    args = parser.parse_args()
    token = os.getenv("CURRENCY_DEVICE_TOKEN", "")
    base = args.url.rstrip("/")
    if not token or (not base.startswith("https://") and not base.startswith("http://127.0.0.1:")):
        parser.error("Set CURRENCY_DEVICE_TOKEN and use HTTPS (or local test server)")
    if args.interval < 60:
        parser.error("Poll interval must be at least 60 seconds")
    while True:
        try:
            print(sync_once(base, args.device_id, token, args.state_dir, args.self_test), flush=True)
        except Exception as error:
            print(f"update_failed: {error}", flush=True)
            try:
                current = json.loads((args.state_dir / "current.json").read_text())
                heartbeat(base, args.device_id, token, "failed", current.get("run_id"), current.get("generation"))
            except Exception:
                pass
            if args.once:
                raise SystemExit(1) from error
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
