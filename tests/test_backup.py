import os
import tempfile
import unittest
import hashlib
from pathlib import Path

import boto3
from cryptography.exceptions import InvalidTag
from moto import mock_aws

from scripts.backup_staging import copy_encrypted_object, encrypt_file
from scripts.restore_drill import decrypt_file


class BackupTest(unittest.TestCase):
    def test_encrypted_backup_round_trip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.dump"
            encrypted = root / "source.dump.enc"
            restored = root / "restored.dump"
            source.write_bytes(os.urandom(128 * 1024))
            key = os.urandom(32)
            encrypt_file(source, encrypted, key)
            self.assertNotIn(source.read_bytes()[:100], encrypted.read_bytes())
            decrypt_file(encrypted, restored, key)
            self.assertEqual(restored.read_bytes(), source.read_bytes())
            content = bytearray(encrypted.read_bytes())
            content[-20] ^= 1
            encrypted.write_bytes(content)
            with self.assertRaises(InvalidTag):
                decrypt_file(encrypted, root / "tampered.dump", key)

    def test_private_object_is_replicated_encrypted_and_recoverable(self):
        raw = b"private model or evidence bytes" * 1000
        key = os.urandom(32)
        with mock_aws(), tempfile.TemporaryDirectory() as temporary:
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="currency-source")
            client.create_bucket(Bucket="currency-backup")
            client.put_object(Bucket="currency-source", Key="models/approved/model.onnx", Body=raw)
            item = client.list_objects_v2(Bucket="currency-source")["Contents"][0]
            copied = copy_encrypted_object(client, "currency-source", "currency-backup",
                {"key": item["Key"], "size": item["Size"], "etag": item["ETag"]},
                "staging/backups/test/", temporary, key)
            self.assertEqual(copied["sha256"], hashlib.sha256(raw).hexdigest())
            encrypted_bytes = client.get_object(Bucket="currency-backup", Key=copied["backup_key"])["Body"].read()
            self.assertNotIn(raw[:100], encrypted_bytes)
            encrypted = Path(temporary) / "saved.enc"
            restored = Path(temporary) / "saved.bin"
            encrypted.write_bytes(encrypted_bytes)
            decrypt_file(encrypted, restored, key)
            self.assertEqual(restored.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
