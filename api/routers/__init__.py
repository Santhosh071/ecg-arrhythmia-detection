from .agent import router as agent_router
from .auth import router as auth_router
from .health import router as health_router
from .history import router as history_router
from .iot_stream import router as iot_router
from .prediction import router as prediction_router
from .reports import router as reports_router
from .signal import router as signal_router

__all__ = [
    "agent_router",
    "auth_router",
    "health_router",
    "history_router",
    "iot_router",
    "prediction_router",
    "reports_router",
    "signal_router",
]
