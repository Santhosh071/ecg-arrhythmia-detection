import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


security = HTTPBearer(auto_error=False)


def _urlsafe_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _urlsafe_decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + padding)


def _auth_secret() -> str:
    return os.getenv("APP_AUTH_SECRET", "ecg-arrhythmia-dev-secret")


def _token_ttl_seconds() -> int:
    return int(os.getenv("APP_AUTH_TTL_HOURS", "12")) * 3600


def _demo_users() -> dict[str, dict]:
    return {
        "clinician": {
            "username": os.getenv("APP_CLINICIAN_USERNAME", "clinician"),
            "password": os.getenv("APP_CLINICIAN_PASSWORD", "demo123"),
            "full_name": os.getenv("APP_CLINICIAN_FULL_NAME", "Clinical Reviewer"),
            "role": "clinician",
        },
        "admin": {
            "username": os.getenv("APP_ADMIN_USERNAME", "admin"),
            "password": os.getenv("APP_ADMIN_PASSWORD", "admin123"),
            "full_name": os.getenv("APP_ADMIN_FULL_NAME", "ECG Platform Admin"),
            "role": "admin",
        },
    }


def authenticate_user(username: str, password: str) -> Optional[dict]:
    for user in _demo_users().values():
        if user["username"] == username and user["password"] == password:
            return {
                "username": user["username"],
                "full_name": user["full_name"],
                "role": user["role"],
            }
    return None


def create_access_token(user: dict) -> str:
    payload = {
        "sub": user["username"],
        "full_name": user["full_name"],
        "role": user["role"],
        "exp": int(time.time()) + _token_ttl_seconds(),
    }
    payload_raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_token = _urlsafe_encode(payload_raw)
    signature = hmac.new(_auth_secret().encode("utf-8"), payload_token.encode("utf-8"), hashlib.sha256).digest()
    return f"{payload_token}.{_urlsafe_encode(signature)}"


def verify_access_token(token: str) -> dict:
    try:
        payload_token, signature_token = token.split(".", 1)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Malformed auth token.") from exc

    expected_signature = hmac.new(
        _auth_secret().encode("utf-8"),
        payload_token.encode("utf-8"),
        hashlib.sha256,
    ).digest()

    provided_signature = _urlsafe_decode(signature_token)
    if not hmac.compare_digest(expected_signature, provided_signature):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth token.")

    payload = json.loads(_urlsafe_decode(payload_token).decode("utf-8"))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Auth token expired.")

    return {
        "username": payload.get("sub", "unknown"),
        "full_name": payload.get("full_name", "Unknown User"),
        "role": payload.get("role", "clinician"),
    }


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")
    return verify_access_token(credentials.credentials)
