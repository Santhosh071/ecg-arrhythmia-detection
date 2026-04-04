from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def _payload(request: Request, status_code: int, detail: str, error_code: str) -> dict:
    return {
        "detail": detail,
        "error": {
            "code": error_code,
            "detail": detail,
            "request_id": getattr(request.state, "request_id", "-"),
            "status_code": status_code,
        },
    }


async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=_payload(request, exc.status_code, detail, f"http_{exc.status_code}"),
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled.exception path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=_payload(request, 500, "Internal server error.", "internal_server_error"),
    )
