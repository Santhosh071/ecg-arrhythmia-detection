import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials

from api.auth import require_auth, security, verify_access_token
from api.dependencies import get_agent, get_db
from api.schemas import ReportRequest, ReportResponse
from api.services import generate_session_report

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Report"])
REPORTS_ROOT = Path(os.getenv("REPORTS_PATH", "C:/ecg_arrhythmia/outputs/reports")).resolve()


@router.post("/report/generate", response_model=ReportResponse)
def generate_report(req: ReportRequest, current_user: dict = Depends(require_auth)):
    try:
        agent = get_agent()
        db = get_db()
        result = generate_session_report(agent, req.patient_id, req.session_data, db)
        return ReportResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("/report/generate error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report/download")
def download_report(
    file_path: str = Query(..., description="Absolute path returned by report generation"),
    auth_token: str | None = Query(None, description="Optional auth token for inline report viewing"),
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    token = None
    if credentials is not None and getattr(credentials, "scheme", "").lower() == "bearer":
        token = credentials.credentials
    elif auth_token:
        token = auth_token

    if token is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    verify_access_token(token)

    try:
        candidate = Path(file_path).resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid report path.") from e

    if REPORTS_ROOT not in candidate.parents and candidate != REPORTS_ROOT:
        raise HTTPException(status_code=403, detail="Report path is outside the reports directory.")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Report file not found.")

    return FileResponse(
        path=str(candidate),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{candidate.name}"'},
    )
