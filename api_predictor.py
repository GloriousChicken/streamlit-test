"""
api_predictor.py
Sends an image to the Docker-based prediction API and returns results
in the same format as dummy_predictor.predict().
Falls back to the dummy predictor on connection errors.
"""

import base64
import io

import requests
import streamlit as st
from PIL import Image
from typing import List, Dict

from dummy_predictor import predict as dummy_predict, classify_outlines as dummy_classify

DEFAULT_API_URL = "REDACTED_API_URL"


def predict_api(
    image: Image.Image,
    api_url: str = DEFAULT_API_URL,
    seed: int = None,
    outlines: List[Dict] = None,
) -> List[Dict]:
    """
    Send *image* to the prediction API and return building dicts.

    If *outlines* is provided (list of {x,y,w,h} dicts), they are included
    in the request body so the API classifies those outlines instead of
    running its own detector.

    On any connection / timeout / HTTP error, displays a warning and
    falls back to the dummy predictor so the UI stays functional.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload: Dict = {"image": img_b64}
    if outlines is not None:
        payload["outlines"] = outlines

    try:
        resp = requests.post(
            api_url,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to prediction API — is the Docker container running?")
    except requests.exceptions.Timeout:
        st.error("Prediction API timed out.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"Prediction API returned an error: {exc.response.status_code}")
    except Exception as exc:
        st.error(f"Unexpected error calling prediction API: {exc}")

    st.warning("Falling back to dummy predictor.")
    if outlines is not None:
        return dummy_classify(outlines, seed=seed)
    return dummy_predict(image, seed=seed)
