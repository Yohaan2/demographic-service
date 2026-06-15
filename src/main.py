import time
import cv2
import numpy as np
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST

from src.core.config import settings
from src.core.logging import setup_logging, logger
from src.pipelines.age_gender_pipeline import AgeGenderPipeline
from src.api.schemas.demographics import AnalyzeResponse
from src.metrics.prometheus import (
    get_prometheus_metrics, 
    REQUESTS_TOTAL, 
    PROCESSING_TIME, 
    FACES_DETECTED
)

# Initialize logging framework before creating application
setup_logging()

# Backpressure Control State variables
active_requests = 0
active_requests_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager that handles startup model loading and teardown operations."""
    logger.info("Starting up FastAPI application...")
    
    # Warm up and load deep learning pipelines into memory exactly once at startup
    try:
        app.state.pipeline = AgeGenderPipeline()
        logger.info("Deep learning pipeline modules successfully loaded and warmed up.")
    except Exception as e:
        logger.critical("Failed to load and warm up pipeline during startup lifespan!", error=str(e))
        raise e
        
    yield
    
    # Teardown logic
    logger.info("Shutting down FastAPI application...")
    # Clear pipeline to free memory space
    if hasattr(app.state, "pipeline"):
        del app.state.pipeline
    logger.info("Application successfully shutdown.")


app = FastAPI(
    title="Real-Time Face Demographic Analytics API",
    version="1.0.0",
    description="High-performance video analytics service for age and gender estimation using SCRFD + ByteTrack + MiVOLO",
    lifespan=lifespan
)

# CORS configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def backpressure_middleware(request: Request, call_next):
    """
    HTTP backpressure controller middleware.
    Prevents memory exhaustion and CPU thrashing by rejecting requests 
    if the concurrent execution queue exceeds MAX_PENDING_REQUESTS.
    Applies only to high-computation routes to avoid throttling frontend assets and diagnostics.
    """
    global active_requests
    
    # Apply backpressure control ONLY to the heavy AI pipeline endpoint
    if request.url.path != "/api/v1/analyze":
        return await call_next(request)
        
    with active_requests_lock:
        if active_requests >= settings.MAX_PENDING_REQUESTS:
            logger.warning(
                "Backpressure limit exceeded! Refusing incoming request with HTTP 429",
                current_active=active_requests,
                limit=settings.MAX_PENDING_REQUESTS
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Server is under extremely high load. Concurrent capacity limit reached ({settings.MAX_PENDING_REQUESTS}). Please retry shortly."
                },
                headers={"Retry-After": "2"}
            )
        active_requests += 1
        
    try:
        response = await call_next(request)
        return response
    finally:
        with active_requests_lock:
            active_requests -= 1


@app.post("/api/v1/analyze", response_model=AnalyzeResponse, summary="Analyze video frame for facial demographics")
async def analyze_frame(
    request: Request,
    camera_id: str = Form(..., description="Unique camera identifier"),
    timestamp: int = Form(..., description="Unix timestamp of the frame"),
    image: UploadFile = File(..., description="Frame image file (JPG/PNG)")
):
    """
    Core demographic analytics endpoint.
    Processes a single frame: Detects faces, Tracks them, Aligns crops, 
    estimates age/gender, aggregates temporally and returns immediately.
    """
    status_code = 500
    try:
        # Validate uploaded image format
        if not image.content_type.startswith("image/"):
            status_code = 400
            raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")
            
        # Read file contents and decode into OpenCV NumPy BGR format
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            status_code = 400
            raise HTTPException(status_code=400, detail="Invalid image file or decoding failed.")
            
        # Access the warmed pipeline from application state
        pipeline: AgeGenderPipeline = request.app.state.pipeline
        
        # Process frame and record pipeline execution time metrics
        start_ts = time.time()
        result = pipeline.process_frame(img_bgr, camera_id)
        latency = time.time() - start_ts
        
        # Prometheus Metrics Tracking
        PROCESSING_TIME.observe(latency)
        FACES_DETECTED.observe(len(result["faces"]))
        status_code = 200
        REQUESTS_TOTAL.labels(camera_id=camera_id, status="200").inc()
        
        return result
        
    except HTTPException as he:
        REQUESTS_TOTAL.labels(camera_id=camera_id, status=str(he.status_code)).inc()
        raise he
    except Exception as e:
        logger.exception("Unexpected pipeline failure during frame processing", camera_id=camera_id, error=str(e))
        REQUESTS_TOTAL.labels(camera_id=camera_id, status="500").inc()
        raise HTTPException(status_code=500, detail=f"Internal pipeline failure: {str(e)}")


@app.get("/health", summary="Perform API health evaluation")
async def health_check():
    """Service health endpoint indicating operational readiness."""
    return {"status": "healthy"}


@app.get("/metrics", summary="Fetch Prometheus metrics")
async def metrics_endpoint():
    """Telemetry exposition endpoint formatted for Prometheus scapers."""
    metrics_data, content_type = get_prometheus_metrics()
    return Response(content=metrics_data, media_type=content_type)


# --- Servir archivos estáticos del frontend React compilado ---
try:
    import os
    from fastapi.staticfiles import StaticFiles
    
    # Calculate absolute path to client/dist (which sits outside the src directory)
    static_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "client", "dist"))
    
    if os.path.exists(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
        logger.info(f"Frontend static files successfully mounted from: {static_dir}")
    else:
        logger.warning(f"Static directory not found at: {static_dir}. Make sure you ran 'npm run build' inside client folder.")
except Exception as e:
    logger.error("Error attempting to mount frontend static files", error=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("Launching server directly with Uvicorn...")
    uvicorn.run(
        "src.main.py:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        workers=settings.WORKERS
    )
