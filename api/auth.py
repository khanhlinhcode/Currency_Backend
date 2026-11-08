import os
from datetime import datetime, timedelta, timezone
from jose import jwt

from .config import ENVIRONMENT

SECRET_KEY = os.getenv("SECRET_KEY", "development-only-secret-change-before-production")
if ENVIRONMENT in {"staging", "production"} and len(SECRET_KEY) < 32:
    raise RuntimeError("SECRET_KEY must contain at least 32 characters")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
