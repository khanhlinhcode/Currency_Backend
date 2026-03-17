from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from .auth import ALGORITHM, SECRET_KEY

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
optional_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)


def decode_token_username(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")

        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        return username
    except JWTError as decode_error:
        raise HTTPException(status_code=401, detail="Invalid token") from decode_error


def get_current_user(token: str = Depends(oauth2_scheme)):
    return decode_token_username(token)


def get_optional_current_user(token: Optional[str] = Depends(optional_oauth2_scheme)):
    if not token:
        return None

    try:
        return decode_token_username(token)
    except HTTPException:
        return None
