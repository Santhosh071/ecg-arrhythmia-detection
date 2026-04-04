import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_auth
from api.schemas import SignalQualityRequest, SignalQualityResponse
from src.layer3_models.inference.preprocess import check_signal_quality

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Signal"])


@router.post("/signal/quality", response_model=SignalQualityResponse)
def signal_quality(req: SignalQualityRequest, current_user: dict = Depends(require_auth)):
    try:
        signal = np.array(req.signal, dtype=np.float64)
        result = check_signal_quality(signal, fs=req.fs)
        return SignalQualityResponse(patient_id=req.patient_id, **result)
    except Exception as e:
        logger.error("/signal/quality error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
