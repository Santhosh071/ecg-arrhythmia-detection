import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import require_auth
from api.dependencies import get_db, get_loader
from api.iot_schemas import (
    IoTAnalyzeRequest,
    IoTAnalyzeResponse,
    IoTLiveWindowResponse,
    IoTStatusResponse,
    IoTStreamRequest,
    IoTStreamResponse,
)
from api.services.iot_stream_service import (
    analyze_iot_window,
    get_iot_live_window,
    get_iot_status,
    ingest_iot_stream,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/iot", tags=["IoT"])


@router.post("/stream", response_model=IoTStreamResponse)
def ingest_stream(req: IoTStreamRequest, current_user: dict = Depends(require_auth)):
    try:
        return IoTStreamResponse(**ingest_iot_stream(req))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("/iot/stream error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status/{device_id}", response_model=IoTStatusResponse)
def stream_status(device_id: str, current_user: dict = Depends(require_auth)):
    try:
        return IoTStatusResponse(**get_iot_status(device_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("/iot/status error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/live/{device_id}", response_model=IoTLiveWindowResponse)
def live_window(
    device_id: str,
    window_sec: float = Query(8.0, ge=1.0, le=60.0),
    current_user: dict = Depends(require_auth),
):
    try:
        return IoTLiveWindowResponse(**get_iot_live_window(device_id, window_sec=window_sec))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("/iot/live error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/analyze", response_model=IoTAnalyzeResponse)
def analyze_stream(req: IoTAnalyzeRequest, current_user: dict = Depends(require_auth)):
    try:
        loader = get_loader()
        try:
            db = get_db()
            response = analyze_iot_window(req, loader, db=db)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("IoT analysis persistence skipped: %s", exc)
            response = analyze_iot_window(req, loader, db=None)
        return IoTAnalyzeResponse(**response)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/iot/analyze error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
