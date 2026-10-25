"""
Prediction API Routes
POST /api/predict/image        — CNN classifier + segmentation
POST /api/predict/timeseries   — LSTM model
GET  /api/predict/risk-score   — composite risk score
"""

import time
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from services.prediction_service import prediction_service

router = APIRouter()


@router.post("/image")
async def predict_image(file: UploadFile = File(...)):
    """Run CNN classifier + segmentation on uploaded satellite image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        if file.filename and not file.filename.lower().endswith(('.tif', '.tiff')):
            raise HTTPException(status_code=400, detail="File must be an image (PNG, JPG, TIF)")

    try:
        contents = await file.read()
        result = prediction_service.predict_image(contents)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/timeseries")
async def predict_timeseries(file: UploadFile = File(...)):
    """Run LSTM forecast on uploaded CSV time-series data."""
    if file.filename and not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV format")

    try:
        contents = await file.read()
        result = prediction_service.predict_timeseries(contents)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-score")
async def get_risk_score(
    lat: float = Query(28.6139, description="Latitude"),
    lng: float = Query(77.2090, description="Longitude"),
):
    """Compute composite flood risk score using real weather, hydro, and elevation data."""
    try:
        return await prediction_service.compute_risk_score(lat, lng)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segmentation")
async def predict_segmentation(file: UploadFile = File(...)):
    """Run PyTorch U-Net (ResNet-34) pixel-wise flood segmentation on Sentinel-1 SAR GeoTIFF or image."""
    try:
        contents = await file.read()
        filename = file.filename or "uploaded.tif"
        result = prediction_service.predict_segmentation(contents, filename=filename)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample-s1")
async def get_sample_s1_chips():
    """List preloaded real Sen1Floods11 Sentinel-1 chips available on server."""
    try:
        chips = prediction_service.list_sample_s1_chips()
        return {"samples": chips}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment-sample")
async def predict_segment_sample(chip_filename: str = Query(..., description="Sample chip filename or ID")):
    """Run U-Net segmentation directly on a preloaded Sen1Floods11 sample chip."""
    try:
        t0 = time.time()
        result = prediction_service.predict_sample_s1_chip(chip_filename)
        result["inference_latency_ms"] = round((time.time() - t0) * 1000, 2)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

