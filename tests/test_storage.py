import unittest

from botocore.exceptions import ClientError
from moto import mock_aws

from api import storage
from scripts.archive_run import put_immutable


class StorageTest(unittest.TestCase):
    def test_run_archive_objects_are_immutable(self):
        import tempfile
        from pathlib import Path
        import scripts.archive_run as archive_run

        previous_bucket = archive_run.S3_BUCKET
        previous_region = storage.S3_REGION
        try:
            with mock_aws(), tempfile.TemporaryDirectory() as directory:
                archive_run.S3_BUCKET = "runs-test"
                storage.S3_REGION = "us-east-1"
                storage._s3_client.cache_clear()
                client = storage._s3_client()
                client.create_bucket(Bucket="runs-test")
                path = Path(directory) / "run.json"
                path.write_text('{"status":"diagnostic"}')
                self.assertEqual(put_immutable(client, "staging/runs/id/run.json", path), "uploaded")
                self.assertEqual(put_immutable(client, "staging/runs/id/run.json", path), "already_present")
                path.write_text('{"status":"approved"}')
                with self.assertRaises(RuntimeError):
                    put_immutable(client, "staging/runs/id/run.json", path)
        finally:
            archive_run.S3_BUCKET = previous_bucket
            storage.S3_REGION = previous_region
            storage._s3_client.cache_clear()

    def test_private_s3_round_trip_and_signed_url(self):
        original = (
            storage.STORAGE_BACKEND, storage.S3_BUCKET,
            storage.S3_REGION, storage.S3_ENDPOINT_URL,
        )
        try:
            with mock_aws():
                storage.STORAGE_BACKEND = "s3"
                storage.S3_BUCKET = "currency-test"
                storage.S3_REGION = "us-east-1"
                storage.S3_ENDPOINT_URL = ""
                storage._s3_client.cache_clear()
                client = storage._s3_client()
                client.create_bucket(Bucket="currency-test")

                storage.save_image("predictions/test.jpg", b"jpeg-data")
                content = client.get_object(Bucket="currency-test", Key="predictions/test.jpg")["Body"].read()
                self.assertEqual(content, b"jpeg-data")
                self.assertIn("Signature=", storage.image_url("predictions/test.jpg"))

                storage.delete_image("predictions/test.jpg")
                with self.assertRaises(ClientError):
                    client.get_object(Bucket="currency-test", Key="predictions/test.jpg")
        finally:
            (
                storage.STORAGE_BACKEND, storage.S3_BUCKET,
                storage.S3_REGION, storage.S3_ENDPOINT_URL,
            ) = original
            storage._s3_client.cache_clear()


if __name__ == "__main__":
    unittest.main()
