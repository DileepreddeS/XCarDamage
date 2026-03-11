# main.py
# FastAPI application — the production-ready API
# Endpoints: /health, /analyze, /heatmap/{id}, /annotated/{id}

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import cv2
import numpy as np
import uuid
import os
import time

from model import load_model, run_detection
from severity import compute_severity
from explainability import generate_heatmap
from schemas import AnalysisResponse, HealthResponse

# ── App Setup ──────────────────────────────────────────────
app = FastAPI(
    title="XCarDamage API",
    description="Explainable Vehicle Damage Detection & Severity Estimation — Trained on VehiDE",
    version="1.0.0",
    docs_url="/docs",
)

# Allow React frontend on port 3000 to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

CLASS_NAMES = [
    "dirty_stain", "missing_parts", "dent",
    "scratch_crack", "puncture", "paint_damage", "broken_glass"
]

# ── Load model on startup ──────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("🚀 XCarDamage API starting...")
    load_model()
    print("✅ API ready!")

# ── Routes ─────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    """Check API is running and model is loaded."""
    return {
        "status":  "healthy",
        "model":   "YOLO11m trained on VehiDE",
        "dataset": "VehiDE — 13,945 images, 7 classes",
        "classes": CLASS_NAMES,
    }

@app.post("/analyze")
async def analyze_damage(file: UploadFile = File(...)):
    """
    Main endpoint — analyze a car image for damage.
    Returns detections with severity scores and heatmap URLs.
    """

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded image
    img_id   = str(uuid.uuid4())[:8]
    img_path = os.path.join(UPLOAD_DIR, f"{img_id}.jpg")

    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    image    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    cv2.imwrite(img_path, image)
    start_time = time.time()

    # Run YOLO detection
    raw_detections = run_detection(img_path)

    # Compute severity for each detection
    detections = []
    for det in raw_detections:
        severity_score, severity_label, features = compute_severity(
            image,
            det["bbox"],
            det["confidence"],
            det["class_name"],
        )
        detections.append({
            "damage_type":    det["class_name"],
            "confidence":     round(det["confidence"], 3),
            "severity_score": severity_score,
            "severity_label": severity_label,
            "bbox":           det["bbox"],
            "class_name":     det["class_name"],
            "explanation":    features,
        })

    # Generate heatmap and annotated image
    heatmap_path   = os.path.join(UPLOAD_DIR, f"{img_id}_heatmap.jpg")
    annotated_path = os.path.join(UPLOAD_DIR, f"{img_id}_annotated.jpg")
    generate_heatmap(image, detections, heatmap_path, annotated_path)

    # Overall severity = worst single damage
    scores        = [d["severity_score"] for d in detections]
    overall_score = round(max(scores), 1) if scores else 0.0
    overall_label = (
        "Severe"    if overall_score >= 65 else
        "Moderate"  if overall_score >= 30 else
        "Minor"     if overall_score >   0 else
        "No Damage"
    )

    processing_ms = round((time.time() - start_time) * 1000, 1)

    return JSONResponse({
        "image_id":          img_id,
        "processing_time_ms": processing_ms,
        "total_damages":     len(detections),
        "overall_severity":  overall_label,
        "overall_score":     overall_score,
        "detections":        detections,
        "heatmap_url":       f"/heatmap/{img_id}",
        "annotated_url":     f"/annotated/{img_id}",
    })

@app.get("/heatmap/{img_id}")
def get_heatmap(img_id: str):
    """Return the Grad-CAM heatmap for an analyzed image."""
    path = os.path.join(UPLOAD_DIR, f"{img_id}_heatmap.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Heatmap not found")
    return FileResponse(path, media_type="image/jpeg")

@app.get("/annotated/{img_id}")
def get_annotated(img_id: str):
    """Return the annotated detection image."""
    path = os.path.join(UPLOAD_DIR, f"{img_id}_annotated.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Annotated image not found")
    return FileResponse(path, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)