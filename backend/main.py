from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn, cv2, numpy as np, uuid, os, time
from PIL import Image
from model import run_detection
from severity import compute_severity
from explainability import generate_heatmap

app = FastAPI(
    title="XCarDamage API",
    description="Explainable Vehicle Damage Detection & Severity Estimation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# PREPROCESSING HELPERS
# ─────────────────────────────────────────────

def get_conf_threshold(image_path: str) -> float:
    """
    Dynamically lower confidence threshold for large images
    where damage occupies a smaller percentage of the frame.
    """
    try:
        img = Image.open(image_path)
        w, h = img.size
        if max(w, h) > 1500:
            return 0.10   # large image → very small damage → lower threshold
        elif max(w, h) > 800:
            return 0.13   # medium-large image
        else:
            return 0.15   # standard image
    except Exception:
        return 0.15


def preprocess_image(image_path: str) -> str:
    """
    Resize oversized images to max 1280px on the longest side.
    This dramatically speeds up inference and improves detection
    of damage that appears tiny in large photos.
    Returns the (possibly new) image path.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        max_dim = max(w, h)

        if max_dim > 1280:
            ratio = 1280 / max_dim
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            img.save(image_path, quality=95)
            print(f"  ↳ Resized from {w}×{h} → {new_w}×{new_h}")
        else:
            print(f"  ↳ Image size OK: {w}×{h} — no resize needed")

    except Exception as e:
        print(f"  ↳ Preprocess warning: {e}")

    return image_path


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "YOLO11m — trained on VehiDE",
        "classes": [
            "dirty_stain", "missing_parts", "dent",
            "scratch_crack", "puncture", "paint_damage", "broken_glass"
        ]
    }


@app.post("/analyze")
async def analyze_damage(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    # Save uploaded image
    img_id = str(uuid.uuid4())[:8]
    img_path = f"{UPLOAD_DIR}/{img_id}.jpg"

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_raw is None:
        raise HTTPException(400, "Could not read image — make sure it is a valid JPG/PNG")

    cv2.imwrite(img_path, image_raw)

    # ── Preprocess ──────────────────────────────
    print(f"\n🔍 Analyzing image {img_id}...")
    img_path = preprocess_image(img_path)
    conf_threshold = get_conf_threshold(img_path)
    print(f"  ↳ Using confidence threshold: {conf_threshold}")

    # Reload (possibly resized) image for heatmap generation
    image = cv2.imread(img_path)

    start_time = time.time()

    # ── Detection ────────────────────────────────
    detections_raw = run_detection(img_path, conf_threshold=conf_threshold)
    detections = []

    for det in detections_raw:
        severity_score, severity_label, features = compute_severity(
            image, det["bbox"], det["confidence"], det["class_name"]
        )
        detections.append({
            "damage_type":    det["class_name"],
            "confidence":     float(round(det["confidence"], 3)),
            "severity_score": float(severity_score),
            "severity_label": severity_label,
            "bbox":           [int(v) for v in det["bbox"]],
            "explanation": {k: float(v) for k, v in features.items()}
        })

    # ── Heatmap ──────────────────────────────────
    heatmap_path   = f"{UPLOAD_DIR}/{img_id}_heatmap.jpg"
    annotated_path = f"{UPLOAD_DIR}/{img_id}_annotated.jpg"
    generate_heatmap(image, detections_raw, heatmap_path, annotated_path)

    # ── Overall assessment ───────────────────────
    scores = [d["severity_score"] for d in detections]
    overall_score = float(round(max(scores), 1)) if scores else 0.0
    overall_label = (
        "Severe"    if overall_score >= 65 else
        "Moderate"  if overall_score >= 30 else
        "Minor"     if overall_score >  0  else
        "No Damage"
    )

    processing_ms = round((time.time() - start_time) * 1000, 1)
    print(f"  ↳ Done in {processing_ms}ms — {len(detections)} damage(s) found")

    return JSONResponse({
        "image_id":          img_id,
        "processing_time_ms": processing_ms,
        "total_damages":     len(detections),
        "overall_severity":  overall_label,
        "overall_score":     overall_score,
        "detections":        detections,
        "heatmap_url":       f"/heatmap/{img_id}",
        "annotated_url":     f"/annotated/{img_id}"
    })


@app.get("/heatmap/{img_id}")
def get_heatmap(img_id: str):
    path = f"{UPLOAD_DIR}/{img_id}_heatmap.jpg"
    if not os.path.exists(path):
        raise HTTPException(404, "Heatmap not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/annotated/{img_id}")
def get_annotated(img_id: str):
    path = f"{UPLOAD_DIR}/{img_id}_annotated.jpg"
    if not os.path.exists(path):
        raise HTTPException(404, "Annotated image not found")
    return FileResponse(path, media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)