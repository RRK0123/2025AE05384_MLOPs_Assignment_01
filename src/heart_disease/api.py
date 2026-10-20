"""Serve the validated fitted pipeline; startup never fits or downloads a model."""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from .dashboard import DASHBOARD_HTML
from .monitoring import EXCLUDED_ROUTES, ServiceMetrics, utc_now
from .predict import feature_frame, load_predictor

LOGGER = logging.getLogger("heart_disease.api")
MAX_BATCH_SIZE = 256
Number = StrictInt | StrictFloat | None


class FeatureRecord(BaseModel):
    """Original UCI fields; null values use the model's fitted imputation."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    age: Number = Field(description="Age in years; positive or null")
    sex: Number = Field(description="Category: 0, 1, or null")
    cp: Number = Field(description="Chest-pain category: 1, 2, 3, 4, or null")
    trestbps: Number = Field(description="Resting blood pressure; positive or null")
    chol: Number = Field(description="Serum cholesterol; positive or null")
    fbs: Number = Field(description="Fasting blood sugar category: 0, 1, or null")
    restecg: Number = Field(description="Resting ECG category: 0, 1, 2, or null")
    thalach: Number = Field(description="Maximum heart rate; positive or null")
    exang: Number = Field(description="Exercise-induced angina: 0, 1, or null")
    oldpeak: Number = Field(description="ST depression; nonnegative or null")
    slope: Number = Field(description="ST slope category: 1, 2, 3, or null")
    ca: Number = Field(description="Major vessels category: 0, 1, 2, 3, or null")
    thal: Number = Field(description="Thal category: 3, 6, 7, or null")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    instances: list[FeatureRecord] = Field(min_length=1, max_length=MAX_BATCH_SIZE)

    @model_validator(mode="after")
    def validate_shared_contract(self):
        # The CLI and API use the same category and measurement validation rules.
        feature_frame(self.model_dump())
        return self


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    prediction: Literal[0, 1]
    probability_disease: float = Field(ge=0, le=1, description="Uncalibrated P(disease=1)")
    confidence: float = Field(ge=0, le=1, description="Probability of the predicted class")


class PredictionResponse(BaseModel):
    model_run_id: str
    predictions: list[Prediction]


def create_app(model_directory: Path | None = None):
    @asynccontextmanager
    async def lifespan(application):
        application.state.metrics = ServiceMetrics()
        configured = model_directory or os.environ.get("HEART_MODEL_DIR")
        try:
            predictor = load_predictor(Path(configured) if configured else None)
        except Exception as error:
            LOGGER.error(
                json.dumps(
                    {
                        "event": "model_load_failed",
                        "level": "ERROR",
                        "timestamp": utc_now(),
                        "error_type": type(error).__name__,
                    }
                )
            )
            raise
        application.state.predictor = predictor
        LOGGER.info(
            json.dumps(
                {
                    "event": "model_loaded",
                    "level": "INFO",
                    "timestamp": utc_now(),
                    "model_run_id": predictor.metadata["run_id"],
                }
            )
        )
        try:
            yield
        finally:
            application.state.predictor = None
            LOGGER.info(
                json.dumps(
                    {
                        "event": "model_unloaded",
                        "level": "INFO",
                        "timestamp": utc_now(),
                        "model_run_id": predictor.metadata["run_id"],
                    }
                )
            )

    application = FastAPI(
        title="Heart Disease Prediction API",
        version="0.2.0",
        lifespan=lifespan,
        description="Predicts recorded disease presence using a saved UCI Cleveland pipeline. "
        "Probabilities are uncalibrated. All 13 fields are required and may be null.",
    )
    application.state.predictor = None
    application.state.metrics = ServiceMetrics()

    @application.middleware("http")
    async def log_request(request: Request, call_next):
        started, request_id = time.perf_counter(), uuid.uuid4().hex
        metrics = application.state.metrics
        counted = request.url.path not in EXCLUDED_ROUTES
        if counted:
            metrics.begin_request()
        error_type = None
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except asyncio.CancelledError:
            error_type, status = "CancelledError", 499
            raise
        except Exception as error:
            error_type = type(error).__name__
            response = JSONResponse(status_code=500, content={"detail": "Prediction service error"})
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            duration = time.perf_counter() - started
            # Framework-owned documentation routes may omit scope['route'].
            # Match only registered paths; never retain arbitrary request URLs.
            known_routes = {getattr(item, "path", None) for item in application.routes}
            route = getattr(request.scope.get("route"), "path", None)
            if route not in known_routes or route is None:
                route = request.url.path if request.url.path in known_routes else "unmatched"
            if counted:
                metrics.finish_request(request.method, route, status, duration)
            predictor = application.state.predictor
            event = {
                "event": "http_request",
                "timestamp": utc_now(),
                "request_id": request_id,
                "method": request.method,
                "route": route,
                "status": status,
                "duration_ms": round(duration * 1000, 3),
                "model_run_id": predictor.metadata["run_id"] if predictor else None,
            }
            if error_type:
                event["error_type"] = error_type
            # Log metadata only: no input records, query strings or prediction values.
            level = (
                logging.ERROR
                if status >= 500
                else logging.WARNING
                if status >= 400
                else logging.INFO
            )
            event["level"] = logging.getLevelName(level)
            LOGGER.log(level, json.dumps(event))

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        # Error context can contain non-JSON values or raw feature records.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {key: item[key] for key in ["loc", "msg", "type"]} for item in error.errors()
                ]
            },
        )

    @application.get("/health", tags=["Health"])
    def health():
        return {"status": "alive"}

    @application.get("/ready", tags=["Health"])
    def ready():
        predictor = application.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return {"status": "ready", "model_run_id": predictor.metadata["run_id"]}

    @application.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
    def predict(payload: PredictionRequest):
        predictor = application.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        started = time.perf_counter()
        try:
            result = PredictionResponse.model_validate(predictor.predict(payload.model_dump()))
        except Exception:
            application.state.metrics.prediction_failed()
            raise
        application.state.metrics.record_predictions(
            [item.prediction for item in result.predictions], time.perf_counter() - started
        )
        return result

    @application.get("/metrics", tags=["Monitoring"], response_class=JSONResponse)
    def metrics():
        """JSON process metrics; health probes and monitoring reads are excluded."""
        predictor = application.state.predictor
        return JSONResponse(
            application.state.metrics.snapshot(predictor.metadata["run_id"] if predictor else None),
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/monitoring", tags=["Monitoring"], response_class=HTMLResponse)
    def dashboard():
        """Self-contained dashboard; no internet services or external assets."""
        return HTMLResponse(DASHBOARD_HTML, headers={"Cache-Control": "no-store"})

    return application


app = create_app()


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    uvicorn.run(create_app(args.model_dir), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
