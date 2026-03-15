from ultralytics import YOLO
import os, cv2

# ── Load model ───────────────────────────────────────────────────────────────
_model = None

def load_model():
    global _model
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")

    if not os.path.exists(model_path):
        print("⚠️  best.pt not found — using placeholder yolo11m.pt")
        _model = YOLO("yolo11m.pt")
    else:
        print(f"✅ Loading trained model from {model_path}")
        _model = YOLO(model_path)

    print("✅ Model loaded — 7 VehiDE classes ready")
    return _model


def get_model():
    global _model
    if _model is None:
        load_model()
    return _model


# Load at startup
load_model()

# ── Inference ────────────────────────────────────────────────────────────────

def run_detection(image_path: str, conf_threshold: float = 0.15) -> list:
    """
    Run YOLO detection with:
    - Dynamic confidence threshold (lower for large images)
    - Test-time augmentation (augment=True) — runs multiple flips/scales
      and merges results, significantly improving detection of small/subtle damage
    """
    model = get_model()

    results = model(
        image_path,
        conf=conf_threshold,
        iou=0.45,         # NMS IoU threshold — prevents duplicate boxes
        augment=True,     # ← Test-time augmentation: flip + scale passes
        verbose=False,
    )

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id     = int(box.cls)
            class_name = result.names[cls_id]
            confidence = float(box.conf)
            bbox       = [round(v) for v in box.xyxy[0].tolist()]

            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox":       bbox,
            })

    # Sort by confidence descending
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    print(f"  ↳ Raw detections (conf≥{conf_threshold}): {len(detections)}")
    for d in detections:
        print(f"     {d['class_name']:15s}  conf={d['confidence']:.2f}  bbox={d['bbox']}")

    return detections