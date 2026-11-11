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

try:
    from .release_signature import verify_payload
except ImportError:  # Direct invocation: python scripts/device_updater.py
    from release_signature import verify_payload


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


CONTRACT_FIELDS = ("class_names", "input_size", "preprocessing",
                   "resize_interpolation", "confidence_threshold")


def validate_contract(contract):
    """A firmware-pinned contract, not a copy of the remote release manifest."""
    if (not isinstance(contract, dict) or contract.get("schema_version") != 1
        or contract.get("channel") not in {"staging", "production"}
        or contract.get("model_format") not in {"onnx", "keras"}
        or not isinstance(contract.get("class_names"), list)
        or not contract["class_names"]
        or any(not isinstance(name, str) or not name for name in contract["class_names"])
        or len(set(contract["class_names"])) != len(contract["class_names"])
        or not isinstance(contract.get("input_size"), list)
        or len(contract["input_size"]) != 2
        or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0
               for size in contract["input_size"])
        or not isinstance(contract.get("preprocessing"), str)
        or not contract["preprocessing"]
        or not isinstance(contract.get("resize_interpolation"), str)
        or not contract["resize_interpolation"]
        or not isinstance(contract.get("confidence_threshold"), (int, float))
        or isinstance(contract["confidence_threshold"], bool)
        or not 0 <= contract["confidence_threshold"] <= 1):
        raise ValueError("Invalid pinned device contract")


def validate_release_contract(release, contract):
    validate_contract(contract)
    if release.get("channel") != contract["channel"]:
        raise ValueError("Release channel differs from pinned device channel")
    if release.get("status") == "no_release":
        return
    manifest = release.get("manifest")
    if (release.get("status") != "available" or not isinstance(manifest, dict)
        or any(manifest.get(field) != contract[field] for field in CONTRACT_FIELDS)
        or not isinstance(release.get("model_filename"), str)
        or Path(release["model_filename"]).suffix != "." + contract["model_format"]
        or manifest.get("release_status") != "approved"
        or manifest.get("sha256") != release.get("model_sha256")
        or manifest.get("version") != release.get("model_version")
        or manifest.get("dataset_version") != release.get("dataset_version")
        or manifest.get("dataset_version") in {None, "legacy-unversioned"}):
        raise ValueError("Release does not match pinned device contract")


def sync_once(base, device_id, token, state_dir, self_test, public_key_base64,
              expected_contract):
    manifest_url = f"{base}/devices/{device_id}/manifest"
    with request(manifest_url, token) as response:
        release = verify_payload(json.load(response), public_key_base64)
    validate_release_contract(release, expected_contract)
    if (not isinstance(release.get("generation"), int) or isinstance(release["generation"], bool)
        or release["generation"] < 0):
        raise ValueError("Invalid release generation")
    pointer_path = state_dir / "current.json"
    current = json.loads(pointer_path.read_text()) if pointer_path.exists() else {}
    if current.get("generation", 0) > release["generation"]:
        raise ValueError("Stale signed release generation")
    if release.get("status") == "no_release":
        return "no_release"
    if not self_test:
        raise ValueError("A local model self-test is required before installation")
    run_id = release["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9-]{1,100}", run_id):
        raise ValueError("Invalid release run ID")
    if release.get("artifact_path") != f"/devices/{device_id}/artifact/{run_id}":
        raise ValueError("Artifact path is not bound to this device and run")
    suffix = Path(release["model_filename"]).suffix
    if suffix not in {".onnx", ".keras"}:
        raise ValueError("Unsupported model file type")
    expected = release["model_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Invalid model digest")
    if (current.get("run_id") == run_id and current.get("model_sha256")
        and current["model_sha256"] != expected):
        raise ValueError("Immutable run ID changed model digest")
    current_model = Path(current.get("model_path", "")) if current.get("model_path") else None
    if (current.get("generation") == release["generation"] and current.get("model_sha256") == expected
        and current_model and current_model.is_file() and sha256(current_model) == expected):
        heartbeat(base, device_id, token, "ready", run_id, release["generation"])
        return "up_to_date"
    # The hash is part of the path, so a failed replacement cannot modify
    # bytes referenced by an older current.json, even if a run ID is reused.
    destination = state_dir / "releases" / run_id / expected
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
    parser.add_argument("--contract", type=Path, required=True,
                        help="Firmware-pinned JSON contract (including channel and model format)")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--self-test", nargs="+", required=True,
                        help="Command and arguments; model path appended")
    args = parser.parse_args()
    token = os.getenv("CURRENCY_DEVICE_TOKEN", "")
    public_key = os.getenv("CURRENCY_RELEASE_PUBLIC_KEY_BASE64", "")
    base = args.url.rstrip("/")
    if (not token or not public_key
        or (not base.startswith("https://") and not base.startswith("http://127.0.0.1:"))):
        parser.error("Set CURRENCY_DEVICE_TOKEN, pin CURRENCY_RELEASE_PUBLIC_KEY_BASE64 and use HTTPS (or local test server)")
    if args.interval < 60:
        parser.error("Poll interval must be at least 60 seconds")
    try:
        expected_contract = json.loads(args.contract.read_text(encoding="utf-8"))
        validate_contract(expected_contract)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"Invalid local device contract: {error}")
    while True:
        try:
            print(sync_once(base, args.device_id, token, args.state_dir, args.self_test,
                            public_key, expected_contract), flush=True)
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
