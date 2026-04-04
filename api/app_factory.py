import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import shutdown, startup
from api.errors import http_exception_handler, unhandled_exception_handler
from api.logging_config import configure_logging
from api.middleware import RequestIdMiddleware
from api.routers import (
    agent_router,
    auth_router,
    health_router,
    history_router,
    iot_router,
    prediction_router,
    reports_router,
    signal_router,
)

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield
    shutdown()



def create_app() -> FastAPI:
    app = FastAPI(
        title="ECG Arrhythmia Detection API",
        description=(
            "AI-powered ECG arrhythmia detection system.\n\n"
            "**DISCLAIMER:** Clinical decision-support tool only. "
            "All outputs require review by a qualified clinician."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestIdMiddleware)

    app.add_exception_handler(Exception, unhandled_exception_handler)
    from fastapi import HTTPException
    app.add_exception_handler(HTTPException, http_exception_handler)

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(prediction_router)
    app.include_router(agent_router)
    app.include_router(reports_router)
    app.include_router(signal_router)
    app.include_router(history_router)
    app.include_router(iot_router)
    return app
