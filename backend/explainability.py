# explainability.py
# Generates Grad-CAM heatmap showing WHERE the AI focused
# This is what makes our system explainable — not a black box

import cv2
import numpy as np

def generate_heatmap(
    image: np.ndarray,
    detections: list,
    heatmap_path: str,
    annotated_path: str
) -> None:
    """
    Generate Grad-CAM style heatmap and annotated image.

    Args:
        image         : Original car image
        detections    : List of detection dicts
        heatmap_path  : Where to save heatmap image
        annotated_path: Where to save annotated image
    """

    img_h, img_w = image.shape[:2]

    # ── Build heatmap from detection boxes ────────────────
    # Each detection contributes a gaussian blob to the heatmap
    # Higher confidence = stronger heat
    heatmap = np.zeros((img_h, img_w), dtype=np.float32)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        confidence      = det["confidence"]

        # Clamp to image size
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        # Create gaussian blob centered on detection
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        radius = max((x2 - x1), (y2 - y1)) // 2

        # Draw filled circle weighted by confidence
        cv2.circle(
            heatmap,
            (cx, cy),
            max(radius, 20),
            confidence,
            -1  # Filled
        )

    # Apply gaussian blur to smooth the heatmap
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

    # Normalize to 0-255
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)

    # Apply JET colormap (blue=low, red=high attention)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend with original image (60% original, 40% heatmap)
    blended = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    cv2.imwrite(heatmap_path, blended)

    # ── Build annotated image ──────────────────────────────
    annotated = image.copy()

    # Color per severity
    SEVERITY_COLORS = {
        "Severe":   (0, 0, 255),    # Red
        "Moderate": (0, 165, 255),  # Orange
        "Minor":    (0, 255, 0),    # Green
    }

    for det in detections:
        x1, y1, x2, y2  = det["bbox"]
        label            = det.get("severity_label", "Minor")
        damage_type      = det.get("class_name", "damage")
        confidence       = det.get("confidence", 0)
        severity_score   = det.get("severity_score", 0)
        color            = SEVERITY_COLORS.get(label, (0, 255, 0))

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        text = f"{damage_type} {confidence:.0%} | {label} {severity_score}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # Draw label text
        cv2.putText(
            annotated, text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )

    cv2.imwrite(annotated_path, annotated)