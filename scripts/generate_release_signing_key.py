"""Generate an Ed25519 release key pair outside Git; never print the private key."""

import argparse
import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[1]


def _create_exclusive(path, data, mode):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as target:
            target.write(data + "\n")
            target.flush()
            os.fsync(target.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def generate(directory, name):
    directory = Path(directory).expanduser().resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError("Signing keys must be stored outside the repository")
    if not name or not all(char.isalnum() or char in "-_" for char in name):
        raise ValueError("Invalid key name")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_file = directory / f"{name}.private.b64"
    public_file = directory / f"{name}.public.b64"
    if private_file.exists() or public_file.exists():
        raise FileExistsError("Key pair already exists; never overwrite an active release key")
    key = Ed25519PrivateKey.generate()
    private = key.private_bytes(encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw, encryption_algorithm=serialization.NoEncryption())
    public = key.public_key().public_bytes(encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)
    _create_exclusive(private_file, base64.b64encode(private).decode("ascii"), 0o600)
    try:
        _create_exclusive(public_file, base64.b64encode(public).decode("ascii"), 0o644)
    except Exception:
        private_file.unlink(missing_ok=True)
        raise
    return private_file, public_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    try:
        private, public = generate(args.directory, args.name)
    except (ValueError, FileExistsError) as error:
        parser.exit(1, str(error) + "\n")
    print(f"Private key file (mode 0600): {private}")
    print(f"Public key file for devices: {public}")


if __name__ == "__main__":
    main()
