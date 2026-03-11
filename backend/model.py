# model.py
# Loads YOLO11m trained on VehiDE and runs detection
# Returns list of detections with class, confidence, bbox

from ultralytics import YOLO
import os

# Class names matching our VehiDE training config
CLASS_NAMES = [
    "dirty_stain",    # 0
    "missing_parts",  # 1
    "dent",           # 2
    "scratch_crack",  # 3
    "puncture",       # 4
    "paint_damage",   # 5
    "broken_glass",   # 6
]

# Global model instance — loaded once when server starts
_model = None

def load_model():
    """Load YOLO11 model — called once at startup."""
    global _model
    model_path = os.path.join(os.path.dirname(__file__), 'best.pt')

    if not os.path.exists(model_path):
        print("⚠️  best.pt not found — using placeholder model for testing")
        print("   Place your trained best.pt in the backend/ folder")
        # Use base YOLO11m for testing until best.pt is ready
        _model = YOLO('yolo11m.pt')
    else:
        print(f"✅ Loading trained model from {model_path}")
        _model = YOLO(model_path)

    print(f"✅ Model loaded — {len(CLASS_NAMES)} classes ready")
    return _model

def get_model():
    """Get model instance — loads if not already loaded."""
    global _model
    if _model is None:
        load_model()
    return _model

def run_detection(image_path: str, conf_threshold: float = 0.25) -> list:
    """
    Run YOLO11 detection on an image.

    Args:
        image_path    : Path to image file
        conf_threshold: Minimum confidence (default 0.25)

    Returns:
        List of dicts with class_name, confidence, bbox
    """
    model = get_model()
    results = model(image_path, conf=conf_threshold, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id     = int(box.cls[0])
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
            confidence = float(box.conf[0])
            bbox       = [round(v) for v in box.xyxy[0].tolist()]

            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox":       bbox,
            })

    return detections