"""Canonical Ed25519 signatures for Internet-delivered device release manifests."""

import base64
import binascii
import hashlib
import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def _decode_key(value, label):
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, TypeError) as error:
        raise ValueError(f"Invalid {label} encoding") from error
    if len(raw) != 32:
        raise ValueError(f"{label} must be 32 raw bytes encoded as base64")
    return raw


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def public_key_base64(private_key_base64):
    private = Ed25519PrivateKey.from_private_bytes(_decode_key(private_key_base64, "private key"))
    raw = private.public_key().public_bytes(encoding=serialization.Encoding.Raw,
                                            format=serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode("ascii")


def sign_payload(payload, private_key_base64):
    """Sign every field except signature, including generation and artifact path."""
    private = Ed25519PrivateKey.from_private_bytes(_decode_key(private_key_base64, "private key"))
    public = private.public_key().public_bytes(encoding=serialization.Encoding.Raw,
                                                format=serialization.PublicFormat.Raw)
    signed = {**payload, "signature_algorithm": "Ed25519",
              "signing_key_id": hashlib.sha256(public).hexdigest()[:16]}
    if "signature" in payload or "signature_algorithm" in payload or "signing_key_id" in payload:
        raise ValueError("Payload already contains signing fields")
    signed["signature"] = base64.b64encode(private.sign(_canonical(signed))).decode("ascii")
    return signed


def verify_payload(payload, public_key_base64):
    """Fail closed for an unsigned or modified release response."""
    public = _decode_key(public_key_base64, "public key")
    if not isinstance(payload, dict) or payload.get("signature_algorithm") != "Ed25519":
        raise ValueError("Signed Ed25519 release manifest required")
    if payload.get("signing_key_id") != hashlib.sha256(public).hexdigest()[:16]:
        raise ValueError("Release signing key does not match pinned device key")
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    try:
        signature = base64.b64decode(payload["signature"], validate=True)
        Ed25519PublicKey.from_public_bytes(public).verify(signature, _canonical(unsigned))
    except (KeyError, TypeError, binascii.Error, InvalidSignature, ValueError) as error:
        raise ValueError("Invalid release manifest signature") from error
    return unsigned
