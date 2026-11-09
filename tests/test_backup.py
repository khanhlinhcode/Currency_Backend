import os
import tempfile
import unittest
from pathlib import Path

from cryptography.exceptions import InvalidTag

from scripts.backup_staging import encrypt_file
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


if __name__ == "__main__":
    unittest.main()
