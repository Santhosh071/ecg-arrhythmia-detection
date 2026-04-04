import logging

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_auth
from api.dependencies import get_db, get_loader
from api.schemas import BatchResultResponse, BeatResultResponse, PredictRequest, SingleBeatRequest
from api.services import predict_batch_result, predict_single_beat_result

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=BatchResultResponse)
def predict_batch(req: PredictRequest, current_user: dict = Depends(require_auth)):
    try:
        loader = get_loader()
        try:
            db = get_db()
            response = predict_batch_result(req, loader, db)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning("Prediction persistence skipped: %s", e)
            response = predict_batch_result(req, loader, db=None)
        return BatchResultResponse(**response)
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("/predict error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/single", response_model=BeatResultResponse)
def predict_single(req: SingleBeatRequest, current_user: dict = Depends(require_auth)):
    try:
        loader = get_loader()
        try:
            result = predict_single_beat_result(req, loader)
            return BeatResultResponse(**result)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("/predict/single error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
