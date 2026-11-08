"""Private S3 compatible storage in production; local files for development."""

import hashlib
import hmac
import re
import time
from functools import lru_cache
from urllib.parse import quote
from botocore.config import Config

from .auth import SECRET_KEY
from .config import (
    ENVIRONMENT, S3_ACCESS_KEY_ID, S3_BUCKET, S3_ENDPOINT_URL, S3_REGION,
    S3_SECRET_ACCESS_KEY, STORAGE_BACKEND, UPLOAD_DIR,
)


@lru_cache(maxsize=1)
def _s3_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL or None,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        config=Config(s3={"addressing_style": "path"}),
    )


def save_image(key: str, content: bytes):
    if STORAGE_BACKEND == "s3":
        _s3_client().put_object(Bucket=S3_BUCKET, Key=key, Body=content, ContentType="image/jpeg")
    elif STORAGE_BACKEND == "local" and ENVIRONMENT == "development":
        target = UPLOAD_DIR / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    else:
        raise RuntimeError("Unsupported storage backend")


def delete_image(key: str):
    if STORAGE_BACKEND == "s3":
        _s3_client().delete_object(Bucket=S3_BUCKET, Key=key)
    else:
        (UPLOAD_DIR / key).unlink(missing_ok=True)


def image_url(key: str):
    if STORAGE_BACKEND == "s3":
        expires = int(time.time()) + 300
        signature = _image_signature(key, expires)
        return f"/images/{quote(key, safe='/')}?expires={expires}&signature={signature}"
    return f"/uploads/{key}"


def _image_signature(key: str, expires: int):
    message = f"{key}\n{expires}".encode()
    return hmac.new(SECRET_KEY.encode(), message, hashlib.sha256).hexdigest()


def valid_image_signature(key: str, expires: int, signature: str):
    if not re.fullmatch(r"predictions/[0-9a-f-]{36}\.jpg", key):
        return False
    now = int(time.time())
    if expires < now or expires > now + 300:
        return False
    return hmac.compare_digest(_image_signature(key, expires), signature)
