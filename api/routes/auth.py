"""内部 API 鉴权（Dashboard 日志查询等）。"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import DEFAULT_INTERNAL_API_KEY

_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    expected = DEFAULT_INTERNAL_API_KEY
    token = None
    if credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    if token is None:
        token = request.query_params.get("api_key")

    if not expected:
        return ""
    if not token or token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid api key",
        )
    return token
