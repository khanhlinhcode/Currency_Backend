import base64
import stat
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.release_signature import public_key_base64, sign_payload, verify_payload
from scripts.generate_release_signing_key import generate


class ReleaseSignatureTest(unittest.TestCase):
    def test_key_generator_uses_exclusive_private_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            private, public = generate(Path(temporary), "staging")
            self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o600)
            self.assertEqual(public.read_text().strip(), public_key_base64(private.read_text().strip()))
            with self.assertRaises(FileExistsError):
                generate(Path(temporary), "staging")

    def test_signed_manifest_is_bound_to_every_field_and_pinned_key(self):
        private = Ed25519PrivateKey.generate()
        private_b64 = base64.b64encode(private.private_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption())).decode()
        public_b64 = public_key_base64(private_b64)
        payload = {"status": "available", "generation": 5, "run_id": "approved-5",
                   "artifact_path": "/devices/abc/artifact/approved-5",
                   "model_sha256": "a" * 64, "manifest": {"version": "2.0.0"}}
        signed = sign_payload(payload, private_b64)
        self.assertEqual(verify_payload(signed, public_b64)["generation"], 5)
        with self.assertRaises(ValueError):
            verify_payload({**signed, "generation": 4}, public_b64)
        with self.assertRaises(ValueError):
            verify_payload(payload, public_b64)
        with self.assertRaises(ValueError):
            verify_payload(signed, base64.b64encode(bytes(32)).decode())


if __name__ == "__main__":
    unittest.main()
