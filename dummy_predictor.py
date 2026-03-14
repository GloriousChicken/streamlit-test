"""
dummy_predictor.py
Fake model that returns random bounding boxes + damage scores.
Swap predict() for the real model with zero changes to app.py.

Output format (list of dicts):
    {
        "x": float,        # bbox left, as fraction of image width  (0-1)
        "y": float,        # bbox top,  as fraction of image height (0-1)
        "w": float,        # bbox width  as fraction of image width  (0-1)
        "h": float,        # bbox height as fraction of image height (0-1)
        "label": int,      # 0=no-damage 1=minor 2=major 3=destroyed
        "confidence": float  # 0-1
    }
"""

import random
from PIL import Image
from typing import List, Dict


DAMAGE_CLASSES = {
    0: "No damage",
    1: "Minor damage",
    2: "Major damage",
    3: "Destroyed",
}

# Realistic class distribution (mirrors xBD challenge)
CLASS_WEIGHTS = [0.68, 0.12, 0.12, 0.08]


def predict(image: Image.Image, seed: int = None) -> List[Dict]:
    """
    Fake inference: generates a grid of building bounding boxes with random damage labels.

    Args:
        image: PIL Image (post-disaster satellite image)
        seed:  optional random seed for reproducibility

    Returns:
        list of building dicts with x, y, w, h, label, confidence
    """
    if seed is not None:
        random.seed(seed)

    W, H = image.size
    buildings = []

    # Grid parameters — mimic urban building density
    cols = random.randint(5, 9)
    rows = random.randint(4, 7)

    cell_w = 1.0 / cols
    cell_h = 1.0 / rows

    for row in range(rows):
        for col in range(cols):
            # Skip some cells randomly (not every cell has a building)
            if random.random() < 0.25:
                continue

            # Place building within cell with some padding + jitter
            pad_x = cell_w * 0.12
            pad_y = cell_h * 0.12
            jitter_x = random.uniform(0, cell_w * 0.08)
            jitter_y = random.uniform(0, cell_h * 0.08)

            bx = col * cell_w + pad_x + jitter_x
            by = row * cell_h + pad_y + jitter_y
            bw = cell_w * random.uniform(0.45, 0.72)
            bh = cell_h * random.uniform(0.45, 0.72)

            # Clip to image bounds
            bx = max(0.0, min(bx, 1.0 - bw))
            by = max(0.0, min(by, 1.0 - bh))

            label = random.choices([0, 1, 2, 3], weights=CLASS_WEIGHTS)[0]
            confidence = round(random.uniform(0.55, 0.99), 2)

            buildings.append({
                "x": round(bx, 4),
                "y": round(by, 4),
                "w": round(bw, 4),
                "h": round(bh, 4),
                "label": label,
                "confidence": confidence,
            })

    return buildings
