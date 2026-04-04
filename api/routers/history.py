import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import require_auth
from api.dependencies import get_db
from api.schemas import AlertsResponse, AlertReviewRequest, AlertReviewResponse, HistoryResponse, ReportsHistoryResponse
from api.services import (
    get_patient_alerts,
    get_patient_history,
    get_patient_reports,
    get_patient_trend,
    update_patient_alert_review,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["History"])


@router.get("/history/{patient_id}", response_model=HistoryResponse)
def get_history(
    patient_id: str,
    limit: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(require_auth),
):
    try:
        db = get_db()
        result = get_patient_history(db, patient_id, limit)
        return HistoryResponse(**result)
    except Exception as e:
        logger.error("/history error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{patient_id}/alerts", response_model=AlertsResponse)
def get_alerts(
    patient_id: str,
    limit: int = Query(20, ge=1, le=200),
    current_user: dict = Depends(require_auth),
):
    try:
        db = get_db()
        result = get_patient_alerts(db, patient_id, limit)
        return AlertsResponse(**result)
    except Exception as e:
        logger.error("/history/alerts error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/history/{patient_id}/alerts/{alert_id}", response_model=AlertReviewResponse)
def review_alert(
    patient_id: str,
    alert_id: int,
    req: AlertReviewRequest,
    current_user: dict = Depends(require_auth),
):
    if req.patient_id != patient_id:
        raise HTTPException(status_code=400, detail="Patient ID mismatch.")

    allowed_statuses = {"new", "reviewed", "acknowledged", "dismissed"}
    review_status = req.review_status.strip().lower()
    if review_status not in allowed_statuses:
        raise HTTPException(status_code=422, detail="Invalid review status.")

    try:
        db = get_db()
        result = update_patient_alert_review(
            db,
            patient_id=patient_id,
            alert_id=alert_id,
            review_status=review_status,
            reviewer_note=req.reviewer_note,
            reviewed_by=current_user.get("username"),
        )
        return AlertReviewResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("/history/alerts/review error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{patient_id}/reports", response_model=ReportsHistoryResponse)
def get_reports(
    patient_id: str,
    limit: int = Query(20, ge=1, le=200),
    current_user: dict = Depends(require_auth),
):
    try:
        db = get_db()
        result = get_patient_reports(db, patient_id, limit)
        return ReportsHistoryResponse(**result)
    except Exception as e:
        logger.error("/history/reports error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{patient_id}/trend")
def get_trend(
    patient_id: str,
    last_n_sessions: int = Query(5, ge=1, le=50),
    current_user: dict = Depends(require_auth),
):
    try:
        db = get_db()
        return get_patient_trend(db, patient_id, last_n_sessions)
    except Exception as e:
        logger.error("/history/trend error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
