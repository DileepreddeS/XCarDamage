# severity.py
# Core research contribution — annotation-free severity estimation
# Computes a 0-100 severity score using 5 measurable image signals
# NO manually labeled severity data needed — fully automatic

import cv2
import numpy as np

# Class weights — some damage types are inherently more severe
CLASS_SEVERITY_WEIGHTS = {
    "dirty_stain":    0.2,   # Least severe — cosmetic only
    "paint_damage":   0.4,   # Moderate — affects appearance
    "scratch_crack":  0.5,   # Moderate — surface damage
    "dent":           0.6,   # Moderate-high — structural
    "missing_parts":  0.7,   # High — functionality affected
    "puncture":       0.8,   # High — structural integrity
    "broken_glass":   1.0,   # Most severe — safety critical
}

def compute_severity(
    image: np.ndarray,
    bbox: list,
    confidence: float,
    class_name: str
) -> tuple:
    """
    Compute severity score for a single detected damage region.

    Args:
        image     : Full car image (numpy array)
        bbox      : [x1, y1, x2, y2] bounding box
        confidence: YOLO detection confidence 0-1
        class_name: Damage class name

    Returns:
        (severity_score, severity_label, feature_dict)
    """

    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # Clamp bbox to image boundaries
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    # Crop the damaged region
    region = image[y1:y2, x1:x2]

    if region.size == 0:
        return 0.0, "Minor", {}

    # ── Feature 1: Area Ratio ──────────────────────────────
    # How much of the car is damaged?
    # Large damaged area = more severe
    damage_area = (x2 - x1) * (y2 - y1)
    total_area  = img_w * img_h
    area_ratio  = min(damage_area / total_area, 1.0)

    # ── Feature 2: Confidence Score ────────────────────────
    # How sure is YOLO about this damage?
    # High confidence = clear, definite damage
    confidence_norm = confidence  # Already 0-1

    # ── Feature 3: Class Weight ────────────────────────────
    # Broken glass is more severe than dirty stain by definition
    class_weight = CLASS_SEVERITY_WEIGHTS.get(class_name, 0.5)

    # ── Feature 4: Edge Density ────────────────────────────
    # Cracks have sharp irregular edges
    # Minor scratches have smooth edges
    gray_region  = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges        = cv2.Canny(gray_region, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # ── Feature 5: Texture Entropy ─────────────────────────
    # Crumpled metal has high texture irregularity
    # Smooth surface has low entropy
    hist = cv2.calcHist([gray_region], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normalize
    hist = hist[hist > 0]     # Remove zeros
    entropy = -np.sum(hist * np.log2(hist))
    entropy_norm = min(entropy / 8.0, 1.0)  # Normalize to 0-1

    # ── Weighted Combination ───────────────────────────────
    # Weights sum to 1.0
    score = (
        0.30 * area_ratio       +   # Area is most important
        0.20 * confidence_norm  +   # Confidence matters
        0.25 * class_weight     +   # Damage type matters
        0.15 * edge_density     +   # Edge sharpness
        0.10 * entropy_norm         # Texture irregularity
    )

    # Scale to 0-100
    severity_score = round(score * 100, 1)

    # ── Severity Label ─────────────────────────────────────
    if severity_score >= 65:
        severity_label = "Severe"
    elif severity_score >= 30:
        severity_label = "Moderate"
    else:
        severity_label = "Minor"

    # Feature breakdown for explainability
    features = {
        "area_ratio":     round(area_ratio, 4),
        "confidence":     round(confidence_norm, 4),
        "class_weight":   class_weight,
        "edge_density":   round(edge_density, 4),
        "texture_entropy": round(entropy_norm, 4),
    }

    return severity_score, severity_label, features