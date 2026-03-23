"""
api_predictor.py
Sends pre/post images + label JSONs to the Docker-based prediction API
and returns a list of {x, y, w, h, label, confidence} dicts.
"""

import io
import json
import os
import re

import requests
import streamlit as st
from PIL import Image
from typing import List, Dict, Optional
from streamlit.runtime.uploaded_file_manager import UploadedFile

DEFAULT_API_URL = os.environ.get("SATDAMAGE_API_URL", "")

SUBTYPE_MAP = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}


def _parse_bboxes_from_json(raw: dict) -> List[Dict]:
    """Extract normalised {x, y, w, h} bounding boxes from an xBD JSON dict."""
    meta = raw["metadata"]
    W, H = float(meta["width"]), float(meta["height"])
    bboxes = []
    for feat in raw["features"]["xy"]:
        wkt = feat["wkt"]
        coords_str = re.search(r'POLYGON\s*\(\((.+?)\)\)', wkt)
        if not coords_str:
            continue
        pairs = coords_str.group(1).split(",")
        xs, ys = [], []
        for pair in pairs:
            parts = pair.strip().split()
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        bboxes.append({
            "x": min(xs) / W, "y": min(ys) / H,
            "w": (max(xs) - min(xs)) / W, "h": (max(ys) - min(ys)) / H,
        })
    return bboxes


def _img_to_upload_tuple(image: Image.Image, filename: str):
    """Convert a PIL image to a (filename, bytes, mime) tuple for requests."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return (filename, buf, "image/png")


def predict_api(
    post_img: Image.Image,
    pre_img: Optional[Image.Image] = None,
    post_json_file: Optional[UploadedFile] = None,
    pre_json_file: Optional[UploadedFile] = None,
    api_url: str = DEFAULT_API_URL,
    seed: int = None,
    model_key: str = "efficientnet",
) -> List[Dict]:
    """
    Send pre/post images and label JSONs to the prediction API.

    The API expects 4 files via multipart/form-data:
      - pre-disaster image (PNG)
      - post-disaster image (PNG)
      - pre-disaster label (JSON)
      - post-disaster label (JSON)

    Returns list of {x, y, w, h, label, confidence} dicts.
    Returns an empty list if required files are missing or the API errors.
    """
    if not api_url:
        st.error("SATDAMAGE_API_URL environment variable is not set.")
        return [], {}

    # Need all 4 files for the API
    if pre_img is None or post_json_file is None or pre_json_file is None:
        st.error("API requires pre & post images + both JSON labels.")
        return [], {}

    # Parse the post JSON to extract bboxes for the response
    post_json_file.seek(0)
    post_json_raw = json.loads(post_json_file.getvalue())
    bboxes = _parse_bboxes_from_json(post_json_raw)

    # Prepare multipart files
    post_json_file.seek(0)
    pre_json_file.seek(0)

    files = [
        ("files", _img_to_upload_tuple(pre_img, "pre_disaster.png")),
        ("files", _img_to_upload_tuple(post_img, "post_disaster.png")),
        ("files", ("pre_disaster.json", pre_json_file, "application/json")),
        ("files", ("post_disaster.json", post_json_file, "application/json")),
    ]

    try:
        resp = requests.post(api_url, files=files, params={"model": model_key}, timeout=300)
        resp.raise_for_status()
        api_result = resp.json()

        buildings_by_index = {b["index"]: b for b in api_result["buildings"]}
        buildings = []
        for i, bbox in enumerate(bboxes):
            b = buildings_by_index.get(i, {})
            buildings.append({
                "x": bbox["x"],
                "y": bbox["y"],
                "w": bbox["w"],
                "h": bbox["h"],
                "label":      int(b.get("prediction", 0)),
                "confidence": float(b.get("confidence", 0.0)),
            })
        return buildings, api_result.get("report", {})

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to prediction API — is the Docker container running?")
    except requests.exceptions.Timeout:
        st.error("Prediction API timed out.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"Prediction API returned an error: {exc.response.status_code}")
    except Exception as exc:
        st.error(f"Unexpected error calling prediction API: {exc}")

    return [], {}
