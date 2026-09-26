# main.py
# FastAPI REST API for AML-Detect
#
# Threshold ownership:
#   Three thresholds are exposed separately in the /status response:
#       - detector_threshold (autoencoder reconstruction error, positive real)
#       - ensemble_threshold (weighted ensemble score in [0, 1])
#       - mtd_threshold (risk score in [0, 1])
#   The previous version exposed a single `threshold` field, which conflated
#   the three scales and caused the 1360.67 vs 0.5 mismatch. This version
#   exposes all three so each layer's threshold can be read unambiguously.

import sys
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from contextlib import asynccontextmanager

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble.ensemble import EnsembleDetector


# ----------------------------------------------------------------------
# Lifespan
# ----------------------------------------------------------------------

ensemble = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ensemble on startup, release on shutdown."""
    global ensemble
    try:
        print("Loading AML-Detect ensemble...")
        X_test = np.load('data/processed/X_test.npy')
        ensemble = EnsembleDetector(X_test.shape[1], enable_mtd=True)
        print("Ensemble loaded successfully.")
    except Exception as e:
        print(f"Failed to load ensemble: {e}")
        ensemble = None
    yield
    ensemble = None


# Initialize FastAPI
app = FastAPI(
    title="AML-Detect API",
    description="Adversarial Machine Learning Detection for CNI",
    version="1.1.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------
# Request and response models
# ----------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    features: List[float]


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    ensemble_score: float
    pre_norm_score: float
    risk_score: float
    weights: List[float]
    component_scores: List[float]
    risk_components: Dict[str, Any]
    mtd_state: Optional[Dict[str, Any]]


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    status: str
    available_models: List[str]
    weights: List[float]
    detector_threshold: float
    ensemble_threshold: float
    mtd_threshold: float
    mtd_enabled: bool
    mtd_state: Optional[Dict[str, Any]]


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "AML-Detect API is running",
        "version": "1.1.0",
        "status": "healthy" if ensemble else "unhealthy",
    }


@app.get("/health", response_model=Dict[str, str])
async def health():
    """Health check endpoint."""
    return {"status": "healthy" if ensemble else "unhealthy"}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get ensemble status with all three thresholds exposed separately."""
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Ensemble not loaded")

    status = ensemble.get_status()
    return StatusResponse(
        status="ready",
        available_models=status.get('available_models', []),
        weights=status.get('weights', []),
        detector_threshold=status.get('detector_threshold', 0.0),
        ensemble_threshold=status.get('ensemble_threshold', 0.0),
        mtd_threshold=status.get('mtd_threshold', 0.0),
        mtd_enabled=status.get('mtd_enabled', False),
        mtd_state=status.get('mtd_state'),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict anomaly score for given features.

    Args:
        request: Features as list of floats

    Returns:
        PredictionResponse with ensemble score, pre-norm debug score,
        risk score, component scores, risk components and MTD state.
    """
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Ensemble not loaded")

    try:
        X = np.array(request.features)
        ensemble_score, risk, weights, scores, risk_components, mtd_state, debug = \
            ensemble.predict_with_mtd(X)

        # Risk components may contain nested weights tuple; normalise for JSON
        safe_risk_components = {}
        for k, v in risk_components.items():
            if isinstance(v, (int, float)):
                safe_risk_components[k] = float(v)
            elif isinstance(v, (list, tuple)):
                safe_risk_components[k] = [float(x) for x in v]
            else:
                safe_risk_components[k] = v

        return PredictionResponse(
            ensemble_score=float(ensemble_score),
            pre_norm_score=float(debug.get('pre_norm_score', ensemble_score)),
            risk_score=float(risk),
            weights=[float(w) for w in weights],
            component_scores=[float(s) for s in scores],
            risk_components=safe_risk_components,
            mtd_state=mtd_state,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Ensemble not loaded")

    status = ensemble.get_status()
    return {
        "detector_threshold": status.get('detector_threshold', 0.0),
        "ensemble_threshold": status.get('ensemble_threshold', 0.0),
        "mtd_threshold": status.get('mtd_threshold', 0.0),
        "available_models": len(status.get('available_models', [])),
        "mtd_active": status.get('mtd_state', {}).get('active', False)
                       if status.get('mtd_enabled') else False,
        "window_size": status.get('window_size', 0),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)