import os

from .config import ENVIRONMENT

DEFAULT_ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin" if ENVIRONMENT == "development" else "")
DEFAULT_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123" if ENVIRONMENT == "development" else "")
