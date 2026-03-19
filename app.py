"""
app.py — SatDamage Assessment HUD (Streamlit prototype)
Run: streamlit run app.py
"""

import json
import numpy as np
import streamlit as st
from PIL import Image
import base64
import io

from api_predictor import predict_api
from pathlib import Path
import re

SUBTYPE_MAP = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}

def parse_xbd_json(raw: dict, is_pre: bool = False) -> list:
    """Parse xBD-format JSON into list of {x, y, w, h, [label, confidence]} dicts."""
    meta = raw["metadata"]
    W, H = float(meta["width"]), float(meta["height"])
    buildings = []
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
        bld = {
            "x": min(xs) / W, "y": min(ys) / H,
            "w": (max(xs) - min(xs)) / W, "h": (max(ys) - min(ys)) / H,
        }
        if not is_pre:
            sub = feat.get("properties", {}).get("subtype", "no-damage")
            bld["label"] = SUBTYPE_MAP.get(sub, 0)
            bld["confidence"] = 1.0  # ground truth
        buildings.append(bld)
    return buildings

# ── Sample pairs bundled in samples/
_SAMPLES_DIR = Path(__file__).parent / "samples"
MODELS = [
    {"key": "EFFICIENTNET-B0", "api": "efficientnet", "f1": "0.667", "f1w": "0.824", "prec": "—",     "rec": "—",     "auc": "—",     "acc": "0.831", "epoch": "—"},
    {"key": "CNN-4CLASS",      "api": "cnn_concat",   "f1": "0.689", "f1w": "—",     "prec": "—",     "rec": "—",     "auc": "—",     "acc": "—",     "epoch": "—"},
]

SAMPLE_PAIRS = [
    {
        "label": "HURRICANE FLORENCE // 00000268",
        "pre":  _SAMPLES_DIR / "hurricane-florence_00000268_pre_disaster.png",
        "post": _SAMPLES_DIR / "hurricane-florence_00000268_post_disaster.png",
        "pre_json":  _SAMPLES_DIR / "hurricane-florence_00000268_pre_disaster.json",
        "post_json": _SAMPLES_DIR / "hurricane-florence_00000268_post_disaster.json",
        "seed": 268,
    },
    {
        "label": "HURRICANE MATTHEW // 00000318",
        "pre":  _SAMPLES_DIR / "hurricane-matthew_00000318_pre_disaster.png",
        "post": _SAMPLES_DIR / "hurricane-matthew_00000318_post_disaster.png",
        "pre_json":  _SAMPLES_DIR / "hurricane-matthew_00000318_pre_disaster.json",
        "post_json": _SAMPLES_DIR / "hurricane-matthew_00000318_post_disaster.json",
        "seed": 318,
    },
    {
        "label": "MEXICO EARTHQUAKE // 00000095",
        "pre":  _SAMPLES_DIR / "mexico-earthquake_00000095_pre_disaster.png",
        "post": _SAMPLES_DIR / "mexico-earthquake_00000095_post_disaster.png",
        "pre_json":  _SAMPLES_DIR / "mexico-earthquake_00000095_pre_disaster.json",
        "post_json": _SAMPLES_DIR / "mexico-earthquake_00000095_post_disaster.json",
        "seed": 95,
    },
    {
        "label": "PALU TSUNAMI // 00000158",
        "pre":  _SAMPLES_DIR / "palu-tsunami_00000158_pre_disaster.png",
        "post": _SAMPLES_DIR / "palu-tsunami_00000158_post_disaster.png",
        "pre_json":  _SAMPLES_DIR / "palu-tsunami_00000158_pre_disaster.json",
        "post_json": _SAMPLES_DIR / "palu-tsunami_00000158_post_disaster.json",
        "seed": 158,
    },
    {
        "label": "SANTA ROSA WILDFIRE // 00000242",
        "pre":  _SAMPLES_DIR / "santa-rosa-wildfire_00000242_pre_disaster.png",
        "post": _SAMPLES_DIR / "santa-rosa-wildfire_00000242_post_disaster.png",
        "pre_json":  _SAMPLES_DIR / "santa-rosa-wildfire_00000242_pre_disaster.json",
        "post_json": _SAMPLES_DIR / "santa-rosa-wildfire_00000242_post_disaster.json",
        "seed": 242,
    },
]

# ── Page config
st.set_page_config(
    page_title="SatDamage // Assessment HUD",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Streamlit chrome
st.markdown("""
<style>
  #MainMenu, footer, header {visibility:hidden}

  /* === DARK ANIMATED BACKGROUND === */
  .stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: #05080d !important;
  }
  @keyframes bgScroll {
    from { background-position: 0 0, 0 0; }
    to   { background-position: 40px 40px, 40px 40px; }
  }
  .stApp {
    background-image:
      linear-gradient(rgba(0,80,160,0.055) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,80,160,0.055) 1px, transparent 1px) !important;
    background-size: 40px 40px !important;
    animation: bgScroll 28s linear infinite !important;
  }

  .block-container { padding-top:.8rem; padding-bottom:.8rem; background:transparent !important; }

  /* === COMPACT UPLOADERS === */
  [data-testid="stFileUploaderDropzone"] {
    background: rgba(5,12,20,0.92) !important;
    border: 1px solid #0a2a4a !important;
    border-radius: 0 !important;
    padding: 5px 10px !important;
    min-height: 60px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 4px !important;
  }
  [data-testid="stFileUploaderDropzone"]:hover {
    border-color: #40c8ff !important;
    box-shadow: 0 0 8px #00aaff33 !important;
  }
  /* hide drag icon + "drag and drop" text + file limit line */
  [data-testid="stFileUploaderDropzone"] svg { display:none !important; }
  [data-testid="stFileUploaderDropzone"] small { display:none !important; }
  [data-testid="stFileUploaderDropzone"] span  { display:none !important; }
  /* injected label */
  [data-testid="stFileUploaderDropzone"]::before {
    content: '[ DRAG FILES HERE ]';
    display: block;
    font-family: monospace !important;
    font-size: 12px !important;
    letter-spacing: 1.5px !important;
    color: #40c8ff !important;
    white-space: nowrap !important;
  }
  /* keep the Browse button visible */
  [data-testid="stFileUploaderDropzone"] button {
    font-family: monospace !important;
    font-size: 12px !important;
    letter-spacing: 1.5px !important;
    color: #40c8ff !important;
    background: transparent !important;
    border: 0.5px solid #0a4a7a !important;
    border-radius: 0 !important;
    padding: 3px 10px !important;
  }
  /* uploaded file name row */
  [data-testid="stFileUploaderFile"] {
    background: rgba(0,20,40,0.8) !important;
    border: 0.5px solid #0a3050 !important;
    border-radius: 0 !important;
    font-family: monospace !important;
    font-size: 12px !important;
    color: #40c8ff !important;
  }

  /* caption text below uploaders */
  p[data-testid="stCaptionContainer"] {
    color: #40c8ff !important;
    font-family: monospace !important;
    letter-spacing: 1.5px !important;
    font-size: 12px !important;
    margin-top: 2px !important;
  }

  /* expander */
  div[data-testid="stExpander"] {
    background: rgba(5,12,20,0.92);
    border: 0.5px solid #0a2a4a !important;
    border-radius: 0 !important;
  }
  div[data-testid="stExpander"] summary {
    color: #40c8ff !important; font-family: monospace; letter-spacing: 1px;
  }
</style>

<!-- fixed scanline overlay over entire page -->
<div style="position:fixed;inset:0;pointer-events:none;z-index:9999;
     background:repeating-linear-gradient(
       transparent 0px,transparent 3px,
       rgba(0,0,8,0.13) 3px,rgba(0,0,8,0.13) 4px)"></div>
""", unsafe_allow_html=True)


# ── Helpers
def pil_to_b64(img: Image.Image, max_w: int = 600) -> str:
    ratio = max_w / img.width if img.width > max_w else 1.0
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_hud(pre_b64: str | None, post_b64: str,
              buildings: list, pre_buildings: list,
              event_name: str = "—",
              gt_buildings: list | None = None,
              report: dict | None = None) -> str:

    buildings_json = json.dumps(buildings)
    pre_buildings_json = json.dumps(pre_buildings)
    gt_buildings_json = json.dumps(gt_buildings or [])

    report_html = ""
    if report:
        cls_colors = {"no-damage":"#00ff88","minor-damage":"#ffdd00","major-damage":"#ff6600","destroyed":"#ff2222"}
        rows = ""
        for cls in ["no-damage", "minor-damage", "major-damage", "destroyed"]:
            if cls in report:
                r = report[cls]
                rows += f"""
            <div class="sdrow" style="align-items:center;gap:0">
                <span style="font-size:15px;white-space:nowrap;color:{cls_colors[cls]};text-shadow:0 0 8px currentColor">F1 SCORE</span>
                <span style="font-size:15px;white-space:nowrap;color:{cls_colors[cls]};text-shadow:0 0 8px currentColor">{r['f1-score']:.2f}</span>
            </div>"""
        report_html = f"""
      <div class="sdp">
        <div class="sdl">PER-CLASS REPORT</div>
        {rows}
      </div>"""

    pre_html = f"""
      <div id="pre-imgbox">
        <img id="pre-sdimg" src="data:image/png;base64,{pre_b64}"
             style="max-width:100%;max-height:420px;width:auto;height:auto;display:block;opacity:.85"/>
        <canvas id="pre-overlay" style="position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair"></canvas>
        <div class="corner tl"></div><div class="corner tr"></div>
        <div class="corner bl"></div><div class="corner br"></div>
        <div id="pre-lockon" class="lockon">
          <div class="lo-corner lo-tl"></div><div class="lo-corner lo-tr"></div>
          <div class="lo-corner lo-bl"></div><div class="lo-corner lo-br"></div>
        </div>
        <div id="pre-tip">
          <div class="tip-hdr">
            <span>STRUCTURE ID</span><span id="pre-tip-id">#0000</span>
          </div>
          <div class="tip-body">
            <div class="tip-lbl" id="pre-tip-lbl">INTACT</div>
            <div class="tip-row">
              <span class="tip-key">STATUS</span>
              <span class="tip-val" id="pre-tip-status">PRE-DISASTER</span>
            </div>
          </div>
        </div>
      </div>""" if pre_b64 else """
      <div style="background:#060e18;border:0.5px solid #0a2a4a;min-height:110px;
           display:flex;align-items:center;justify-content:center;
           color:#0a2a4a;font-size:10px;letter-spacing:2.5px">NO FEED</div>"""

    return f"""
<style>
  /* === BASE === */
  * {{box-sizing:border-box;margin:0;padding:0}}

  @keyframes bgScroll {{
    from {{ background-position:0 0; }}
    to   {{ background-position:40px 40px; }}
  }}
  body {{
    background-color:#05080d;
    background-image:
      linear-gradient(rgba(0,80,160,0.04) 1px,transparent 1px),
      linear-gradient(90deg,rgba(0,80,160,0.04) 1px,transparent 1px);
    background-size:40px 40px;
    animation:bgScroll 28s linear infinite;
    font-family:'Courier New',monospace;color:#40c8ff;
  }}

  /* === PANEL === */
  .sdp {{
    background:rgba(5,12,20,0.88);
    border:0.5px solid #0a2a4a;
    clip-path:polygon(0 0,calc(100% - 12px) 0,100% 12px,100% 100%,0 100%);
    box-shadow:inset 0 0 0 1px #0a4a7a,0 0 10px #00aaff22;
    padding:10px 12px;
  }}

  /* === SECTION LABELS === */
  .sdl {{
    font-size:13px;color:#40c8ff;letter-spacing:2.5px;text-transform:uppercase;
    margin-bottom:8px;padding-bottom:5px;border-bottom:1px solid #0a3050;
  }}

  /* === LAYOUT === */
  .sdgrid {{display:grid;grid-template-columns:1fr 1fr 1fr 210px;gap:8px}}
  .sdcol  {{display:flex;flex-direction:column;gap:8px}}
  .sdrow  {{
    display:flex;justify-content:space-between;align-items:center;gap:8px;
    padding:3px 0;font-size:14px;border-bottom:1px solid #060e18;
  }}
  .sdrow span:first-child {{color:#40c8ff;font-size:13px;letter-spacing:1px}}
  .sdrow span:last-child  {{color:#40c8ff;font-size:14px;font-weight:600}}

  /* === HEADER === */
  .sdheader {{
    display:flex;justify-content:space-between;align-items:center;
    padding:8px 14px;margin-bottom:0;
  }}
  .sdheader .title {{
    font-size:18px;font-weight:700;letter-spacing:3px;
    color:#40c8ff;text-shadow:0 0 8px #00aaff88;
  }}
  .sdheader .sub {{font-size:13px;color:#40c8ff;letter-spacing:1px;margin-bottom:5px}}


  /* === PROGRESS BAR === */
  #pbar-track {{
    height:2px;background:#060e18;margin:6px 0 8px;overflow:hidden;position:relative;
  }}
  #pbar {{
    position:absolute;height:100%;width:0%;
    background:linear-gradient(90deg,#0055aa,#00ff88);
    box-shadow:0 0 8px #00ff8866;
    transition:width .05s linear;
  }}
  @keyframes pbarPulse {{
    0%,100% {{box-shadow:0 0 6px #00ff8866}}
    50%     {{box-shadow:0 0 16px #00ff88cc}}
  }}
  #pbar.done {{animation:pbarPulse 1.5s ease-in-out 3}}

  /* === IMAGE BOXES === */
  .img-wrapper  {{width:100%;text-align:center}}
  .imgbox-static {{
    display:inline-block;position:relative;line-height:0;max-width:100%;
    border:0.5px solid #0a3050;overflow:hidden;
  }}
  #imgbox {{
    display:inline-block;position:relative;line-height:0;max-width:100%;
    border:0.5px solid #0a3050;overflow:hidden;
  }}
  #pre-imgbox {{
    display:inline-block;position:relative;line-height:0;max-width:100%;
    border:0.5px solid #0a3050;overflow:hidden;
  }}
  #gt-imgbox {{
    display:inline-block;position:relative;line-height:0;max-width:100%;
    border:0.5px solid #0a3050;overflow:hidden;
  }}

  /* Corner brackets */
  .corner {{position:absolute;width:14px;height:14px;z-index:5;pointer-events:none;opacity:0.5}}
  .corner.tl {{top:4px;left:4px;border-top:2px solid #40c8ff;border-left:2px solid #40c8ff}}
  .corner.tr {{top:4px;right:4px;border-top:2px solid #40c8ff;border-right:2px solid #40c8ff}}
  .corner.bl {{bottom:4px;left:4px;border-bottom:2px solid #40c8ff;border-left:2px solid #40c8ff}}
  .corner.br {{bottom:4px;right:4px;border-bottom:2px solid #40c8ff;border-right:2px solid #40c8ff}}
  @keyframes cornerSlideIn {{
    0%  {{opacity:0;transform:scale(1.5)}}
    60% {{opacity:1}}
    100%{{opacity:0.7;transform:scale(1)}}
  }}
  .corner.scanned {{animation:cornerSlideIn .4s ease-out forwards}}

  /* === LOCK-ON === */
  .lockon {{position:absolute;pointer-events:none;display:none;z-index:8}}
  .lo-corner {{position:absolute;width:8px;height:8px}}
  .lo-tl {{top:0;left:0;border-top:2px solid #00ff88;border-left:2px solid #00ff88}}
  .lo-tr {{top:0;right:0;border-top:2px solid #00ff88;border-right:2px solid #00ff88}}
  .lo-bl {{bottom:0;left:0;border-bottom:2px solid #00ff88;border-left:2px solid #00ff88}}
  .lo-br {{bottom:0;right:0;border-bottom:2px solid #00ff88;border-right:2px solid #00ff88}}
  @keyframes lockOnAnim {{
    0%   {{transform:scale(1.8);opacity:0}}
    50%  {{opacity:1}}
    100% {{transform:scale(1);opacity:1}}
  }}
  .lockon.active {{display:block;animation:lockOnAnim .2s ease-out forwards}}

  /* === TOOLTIP === */
  #pre-tip {{
    position:absolute;display:none;pointer-events:none;z-index:20;
    background:rgba(0,12,25,0.97);border:1px solid #0a4a7a;
    clip-path:polygon(0 0,calc(100% - 8px) 0,100% 8px,100% 100%,0 100%);
    box-shadow:0 0 14px #00aaff44;min-width:180px;
  }}
  #pre-lockon .lo-corner {{ border-color: #40c8ff }}
  #gt-tip {{
    position:absolute;display:none;pointer-events:none;z-index:20;
    background:rgba(0,12,25,0.97);border:1px solid #0a4a7a;
    clip-path:polygon(0 0,calc(100% - 8px) 0,100% 8px,100% 100%,0 100%);
    box-shadow:0 0 14px #00aaff44;min-width:180px;
  }}
  #sdtip {{
    position:absolute;display:none;pointer-events:none;z-index:20;
    background:rgba(0,12,25,0.97);border:1px solid #0a4a7a;
    clip-path:polygon(0 0,calc(100% - 8px) 0,100% 8px,100% 100%,0 100%);
    box-shadow:0 0 14px #00aaff44;min-width:180px;
  }}
  .tip-hdr {{
    display:flex;justify-content:space-between;align-items:center;
    padding:4px 8px;border-bottom:1px solid #0a3050;
    font-size:12px;letter-spacing:1.5px;color:#40c8ff;
    white-space:nowrap;gap:10px;
  }}
  .tip-body {{padding:5px 8px}}
  .tip-lbl {{font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:6px}}
  .tip-row {{display:flex;justify-content:space-between;align-items:center;gap:12px;white-space:nowrap}}
  .tip-key {{font-size:12px;color:#40c8ff;letter-spacing:1px}}
  .tip-val {{font-size:14px;font-weight:600;color:#40c8ff}}

  /* === IMG FOOTER === */
  .img-footer {{
    display:flex;justify-content:space-between;
    margin-top:4px;font-size:12px;color:#40c8ff;letter-spacing:1px;
  }}

  /* === DAMAGE LEVEL === */
  #sdtlabel {{font-size:24px;font-weight:700;letter-spacing:3px;margin-bottom:8px;color:#40c8ff}}
  @keyframes critPulse {{
    0%,100%{{text-shadow:0 0 6px currentColor}}
    50%    {{text-shadow:0 0 20px currentColor,0 0 40px currentColor}}
  }}
  #sdtlabel.critical {{animation:critPulse 1s ease-in-out infinite}}
  .seg-bar {{display:flex;gap:3px;margin-bottom:6px;height:7px}}
  .seg {{
    flex:1;background:#060e18;
    clip-path:polygon(2px 0%,100% 0%,calc(100% - 2px) 100%,0% 100%);
    transition:background .08s;
  }}
  #sdtpct {{font-size:12px;color:#40c8ff;letter-spacing:1.5px}}

  /* === DAMAGE DISTRIBUTION === */
  .dist-row  {{margin-bottom:6px}}
  .dist-hdr  {{display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px;letter-spacing:1px}}
  .dist-segs {{display:flex;gap:2px;height:4px}}
  .dist-seg  {{
    flex:1;background:#060e18;
    clip-path:polygon(1px 0%,100% 0%,calc(100% - 1px) 100%,0% 100%);
    transition:background .1s;
  }}

  /* === LEGEND + BUTTON === */
  .legend   {{display:flex;gap:12px;flex-wrap:wrap;align-items:center}}
  .leg-item {{font-size:13px;display:flex;align-items:center;gap:5px;letter-spacing:1px;color:#40c8ff}}
  .leg-sw   {{width:9px;height:9px;clip-path:polygon(0 0,calc(100% - 3px) 0,100% 3px,100% 100%,0 100%)}}
  .sdfooter {{display:flex;justify-content:space-between;align-items:center;margin-top:8px;flex-wrap:wrap;gap:8px}}
  .sc2btn {{
    font-family:'Courier New',monospace;font-size:14px;letter-spacing:2.5px;
    text-transform:uppercase;padding:7px 20px;
    background:rgba(0,25,45,0.9);border:1px solid #0a4a7a;color:#40c8ff;
    clip-path:polygon(0 0,calc(100% - 10px) 0,100% 10px,100% 100%,10px 100%,0 calc(100% - 10px));
    cursor:pointer;box-shadow:0 0 8px #00aaff22;transition:all .2s;
  }}
  .sc2btn:hover {{
    background:rgba(0,55,90,0.95);border-color:#40c8ff;
    box-shadow:0 0 18px #00aaff77,inset 0 0 8px #00335566;
    color:#90e8ff;text-shadow:0 0 8px #40c8ff;
  }}
</style>

<div style="padding:6px 0">

  <!-- HEADER -->
  <div class="sdp sdheader">
    <span class="title">SATDAMAGE // ASSESSMENT HUD</span>
    <div class="sub">{event_name}</div>
  </div>

  <!-- SCAN PROGRESS BAR -->
  <div id="pbar-track"><div id="pbar"></div></div>

  <!-- MAIN GRID: PRE | POST+OVERLAY | STATS -->
  <div class="sdgrid">

    <!-- PRE-DISASTER -->
    <div class="sdp">
      <div class="sdl">PRE-DISASTER</div>
      <div class="img-wrapper">{pre_html}</div>
      <div class="img-footer">
        <span id="pre-coord">X: — &nbsp; Y: —</span>
        <span>PRE-DISASTER</span>
      </div>
    </div>

    <!-- POST-DISASTER + OVERLAY -->
    <div class="sdp">
      <div class="sdl">MODEL PREDICTION</div>
      <div class="img-wrapper">
        <div id="imgbox">
          <img id="sdimg" src="data:image/png;base64,{post_b64}"
               style="max-width:100%;max-height:420px;width:auto;height:auto;display:block;opacity:.85"/>
          <canvas id="sdoverlay"
                  style="position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair"></canvas>
          <div class="corner tl" id="c-tl"></div>
          <div class="corner tr" id="c-tr"></div>
          <div class="corner bl" id="c-bl"></div>
          <div class="corner br" id="c-br"></div>
          <div id="lockon" class="lockon">
            <div class="lo-corner lo-tl"></div><div class="lo-corner lo-tr"></div>
            <div class="lo-corner lo-bl"></div><div class="lo-corner lo-br"></div>
          </div>
          <div id="sdtip">
            <div class="tip-hdr">
              <span>TARGET ACQUIRED</span><span id="tip-id">#0000</span>
            </div>
            <div class="tip-body">
              <div class="tip-lbl" id="tip-lbl">—</div>
              <div class="tip-row">
                <span class="tip-key">CONFIDENCE</span>
                <span class="tip-val" id="tip-conf">—</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="img-footer">
        <span id="sdcoord">X: — &nbsp; Y: —</span>
        <span id="sdstatus">OVERLAY READY</span>
      </div>
    </div>

    <!-- GROUND TRUTH -->
    <div class="sdp">
      <div class="sdl">GROUND TRUTH</div>
      <div class="img-wrapper">
        <div id="gt-imgbox">
          <img id="gt-sdimg" src="data:image/png;base64,{post_b64}"
               style="max-width:100%;max-height:420px;width:auto;height:auto;display:block;opacity:.85"/>
          <canvas id="gt-overlay"
                  style="position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair"></canvas>
          <div class="corner tl"></div><div class="corner tr"></div>
          <div class="corner bl"></div><div class="corner br"></div>
          <div id="gt-lockon" class="lockon">
            <div class="lo-corner lo-tl"></div><div class="lo-corner lo-tr"></div>
            <div class="lo-corner lo-bl"></div><div class="lo-corner lo-br"></div>
          </div>
          <div id="gt-tip">
            <div class="tip-hdr">
              <span>GROUND TRUTH</span><span id="gt-tip-id">#0000</span>
            </div>
            <div class="tip-body">
              <div class="tip-lbl" id="gt-tip-lbl">—</div>
              <div class="tip-row">
                <span class="tip-key">SOURCE</span>
                <span class="tip-val" id="gt-tip-src">GROUND TRUTH</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="img-footer">
        <span id="gt-coord">X: — &nbsp; Y: —</span>
        <span>GROUND TRUTH</span>
      </div>
    </div>

    <!-- STATS COLUMN -->
    <div class="sdcol">
      <div class="sdp">
        <div class="sdl">DAMAGE LEVEL</div>
        <div id="sdtlabel">—</div>
        <div class="seg-bar" id="sdtsegs"></div>
        <div id="sdtpct">run scan first</div>
      </div>
      <div class="sdp">
        <div class="sdl">DAMAGE DISTRIBUTION</div>
        <div id="sddist">
          <div style="font-size:13px;color:#40c8ff;letter-spacing:1.5px">AWAITING SCAN...</div>
        </div>
      </div>
      {report_html}
    </div>
  </div>

  <!-- FOOTER -->
  <div class="sdfooter">
    <div class="legend">
      <span class="leg-item"><span class="leg-sw" style="background:#00cc66"></span>No damage</span>
      <span class="leg-item"><span class="leg-sw" style="background:#ccaa00"></span>Minor</span>
      <span class="leg-item"><span class="leg-sw" style="background:#cc4400"></span>Major</span>
      <span class="leg-item"><span class="leg-sw" style="background:#aa0000"></span>Destroyed</span>
    </div>
    <button class="sc2btn" onclick="runScan()">RUN ASSESSMENT</button>
  </div>

</div>

<script>
const BUILDINGS   = {buildings_json};
const PRE_BUILDINGS = {pre_buildings_json};
const GT_BUILDINGS  = {gt_buildings_json};
const FILLS  = ['#00cc66','#ccaa00','#cc4400','#aa0000'];
const HFILLS = ['#00ff88','#ffdd00','#ff6600','#ff2222'];
const LABELS = ['No damage','Minor damage','Major damage','Destroyed'];
const NSEGS  = 12;
const NDSEG  = 20;

const img    = document.getElementById('sdimg');
const canvas = document.getElementById('sdoverlay');
const ctx    = canvas.getContext('2d');
const TIP    = document.getElementById('sdtip');
const LOCKON = document.getElementById('lockon');
const PBAR   = document.getElementById('pbar');

const preImg    = document.getElementById('pre-sdimg');
const preCanvas = document.getElementById('pre-overlay');
const preCtx    = preCanvas ? preCanvas.getContext('2d') : null;

const PRE_TIP    = document.getElementById('pre-tip');
const PRE_LOCKON = document.getElementById('pre-lockon');
let preHovId = -1;

const gtImg    = document.getElementById('gt-sdimg');
const gtCanvas = document.getElementById('gt-overlay');
const gtCtx    = gtCanvas ? gtCanvas.getContext('2d') : null;
const GT_TIP    = document.getElementById('gt-tip');
const GT_LOCKON = document.getElementById('gt-lockon');
let gtHovId = -1;

let scanned = false;
let hovId   = -1;

// ── Init threat segments
(function() {{
  const bar = document.getElementById('sdtsegs');
  for (let i = 0; i < NSEGS; i++) {{
    const s = document.createElement('div');
    s.className = 'seg'; s.id = 'tseg' + i;
    bar.appendChild(s);
  }}
}})();

// ── Canvas sync

function sync() {{
  const r = img.getBoundingClientRect();
  canvas.width  = Math.round(r.width)  || img.naturalWidth;
  canvas.height = Math.round(r.height) || img.naturalHeight;
}}

function syncPre() {{
  if (!preImg || !preCanvas) return;
  const r = preImg.getBoundingClientRect();
  preCanvas.width  = Math.round(r.width)  || preImg.naturalWidth;
  preCanvas.height = Math.round(r.height) || preImg.naturalHeight;
}}

function drawPre() {{
  if (!preCtx) return;
  preCtx.clearRect(0, 0, preCanvas.width, preCanvas.height);
  const W = preCanvas.width, H = preCanvas.height;
  PRE_BUILDINGS.forEach((b, i) => {{
    const hov = i === preHovId;
    preCtx.fillStyle   = hov ? '#40c8ff44' : '#40c8ff28';
    preCtx.strokeStyle = hov ? '#80e0ff' : '#40c8ff';
    preCtx.lineWidth   = hov ? 1.5 : 0.8;
    preCtx.fillRect(b.x*W, b.y*H, b.w*W, b.h*H);
    preCtx.strokeRect(b.x*W, b.y*H, b.w*W, b.h*H);
  }});
}}

function syncGt() {{
  if (!gtImg || !gtCanvas) return;
  const r = gtImg.getBoundingClientRect();
  gtCanvas.width  = Math.round(r.width)  || gtImg.naturalWidth;
  gtCanvas.height = Math.round(r.height) || gtImg.naturalHeight;
}}

function drawGt() {{
  if (!gtCtx) return;
  gtCtx.clearRect(0, 0, gtCanvas.width, gtCanvas.height);
  const W = gtCanvas.width, H = gtCanvas.height;
  GT_BUILDINGS.forEach((b, i) => {{
    const hov = i === gtHovId;
    const lbl = b.label || 0;
    gtCtx.fillStyle   = (hov ? HFILLS : FILLS)[lbl] + (hov ? '44' : '28');
    gtCtx.strokeStyle = (hov ? HFILLS : FILLS)[lbl];
    gtCtx.lineWidth   = hov ? 1.5 : 0.8;
    gtCtx.fillRect(b.x*W, b.y*H, b.w*W, b.h*H);
    gtCtx.strokeRect(b.x*W, b.y*H, b.w*W, b.h*H);
  }});
}}

function showGtLockOn(b) {{
  GT_LOCKON.style.left=(b.x*100)+'%'; GT_LOCKON.style.top=(b.y*100)+'%';
  GT_LOCKON.style.width=(b.w*100)+'%'; GT_LOCKON.style.height=(b.h*100)+'%';
  GT_LOCKON.className=''; void GT_LOCKON.offsetWidth;
  GT_LOCKON.className='lockon active';
}}

// ── Draw overlays
function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!scanned) return;
  const W = canvas.width, H = canvas.height;
  BUILDINGS.forEach((b, i) => {{
    const x = b.x*W, y = b.y*H, w = b.w*W, h = b.h*H;
    const hov = i === hovId;
    ctx.fillStyle   = (hov ? HFILLS : FILLS)[b.label] + (hov ? '44' : '28');
    ctx.strokeStyle = (hov ? HFILLS : FILLS)[b.label];
    ctx.lineWidth   = hov ? 1.5 : 0.8;
    ctx.fillRect(x,y,w,h); ctx.strokeRect(x,y,w,h);
    if (hov) {{
      ctx.fillStyle = HFILLS[b.label];
      ctx.font = 'bold 10px monospace';
      ctx.fillText(Math.round(b.confidence*100)+'%', x+3, y+10);
    }}
  }});
}}

// ── Stats update
function updateStats() {{
  const cnt = [0,0,0,0];
  BUILDINGS.forEach(b => cnt[b.label]++);
  const tot = cnt.reduce((a,v)=>a+v,0) || 1;

  const dist = document.getElementById('sddist');
  dist.innerHTML = '';
  [0,1,2,3].forEach(i => {{
    const filled = Math.round(cnt[i]/tot * NDSEG);
    let segs = '';
    for (let s=0;s<NDSEG;s++)
      segs += `<div class="dist-seg" style="background:${{s<filled?HFILLS[i]:'#060e18'}}"></div>`;
    dist.innerHTML += `
      <div class="dist-row">
        <div class="dist-hdr">
          <span style="color:${{HFILLS[i]}}">${{LABELS[i]}}</span>
          <span style="color:#40c8ff">${{cnt[i]}}</span>
        </div>
        <div class="dist-segs">${{segs}}</div>
      </div>`;
  }});

  const dpct = Math.round((cnt[1]+cnt[2]+cnt[3])/tot*100);
  const filledSegs = Math.round(dpct/100*NSEGS);
  const [lbl,col] = dpct<10?['LOW','#00ff88']:dpct<25?['MODERATE','#ffdd00']:
                    dpct<45?['HIGH','#ff6600']:['CRITICAL','#ff2222'];
  for (let i=0;i<NSEGS;i++) {{
    const s = document.getElementById('tseg'+i);
    s.style.background = i<filledSegs ? col : '#060e18';
    s.style.boxShadow  = i<filledSegs ? `0 0 4px ${{col}}88` : 'none';
  }}
  const tl = document.getElementById('sdtlabel');
  tl.textContent = lbl; tl.style.color = col;
  tl.style.textShadow = `0 0 10px ${{col}}88`;
  tl.className = lbl==='CRITICAL'?'critical':'';
  document.getElementById('sdtpct').textContent = dpct+'% STRUCTURES AFFECTED';
}}

// ── Scan animation
window.runScan = function() {{
  if (!scanned) {{ scanned = true; sync(); }}
  document.getElementById('sdstatus').textContent = 'SCANNING...';
  PBAR.classList.remove('done');
  PBAR.style.width = '0%';
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const W = canvas.width, H = canvas.height;
  let y = 0;
  const revealed = new Set(), flashTime = {{}};

  (function step() {{
    y += 5;
    ctx.clearRect(0,0,W,H);

    BUILDINGS.forEach((b,i) => {{
      if ((b.y+b.h)*H > y) return;
      if (!revealed.has(i)) {{ revealed.add(i); flashTime[i]=Date.now(); }}
      const bx=b.x*W,by=b.y*H,bw=b.w*W,bh=b.h*H;
      const flash = Date.now()-(flashTime[i]||0) < 220
        ? (1-(Date.now()-(flashTime[i]||0))/220) : 0;
      if (flash>0) {{
        ctx.save(); ctx.globalAlpha=0.3+flash*0.7;
        ctx.fillStyle=HFILLS[b.label]+'cc';
        ctx.fillRect(bx,by,bw,bh); ctx.restore();
      }}
      ctx.fillStyle=FILLS[b.label]+'28';
      ctx.strokeStyle=FILLS[b.label]; ctx.lineWidth=0.8;
      ctx.fillRect(bx,by,bw,bh); ctx.strokeRect(bx,by,bw,bh);
    }});

    // Trailing gradient
    if (y>0) {{
      const tt = Math.max(0,y-H*0.28);
      const g = ctx.createLinearGradient(0,tt,0,y);
      g.addColorStop(0,'rgba(0,255,136,0)');
      g.addColorStop(1,'rgba(0,255,136,0.045)');
      ctx.fillStyle=g; ctx.fillRect(0,tt,W,y-tt);
    }}
    // Scan line
    ctx.save();
    ctx.shadowColor='#00ff88'; ctx.shadowBlur=10;
    ctx.fillStyle='rgba(0,255,136,0.95)';
    ctx.fillRect(0,y-2,W,3);
    ctx.restore();

    // Progress bar
    PBAR.style.width = Math.min(100, y/H*100) + '%';

    if (y < H+10) {{ requestAnimationFrame(step); }}
    else {{
      draw();
      syncPre(); drawPre();
      document.getElementById('sdstatus').textContent =
        'COMPLETE // '+BUILDINGS.length+' STRUCTURES';
      PBAR.style.width = '100%';
      PBAR.classList.add('done');
      updateStats();
      ['c-tl','c-tr','c-bl','c-br'].forEach((id,i) =>
        setTimeout(()=>document.getElementById(id).classList.add('scanned'), i*90));
    }}
  }})();
}};

// ── Pre-image lock-on
function showPreLockOn(b) {{
  PRE_LOCKON.style.left=(b.x*100)+'%'; PRE_LOCKON.style.top=(b.y*100)+'%';
  PRE_LOCKON.style.width=(b.w*100)+'%'; PRE_LOCKON.style.height=(b.h*100)+'%';
  PRE_LOCKON.className=''; void PRE_LOCKON.offsetWidth;
  PRE_LOCKON.className='lockon active';
}}

// ── Lock-on
function showLockOn(b) {{
  LOCKON.style.left=(b.x*100)+'%'; LOCKON.style.top=(b.y*100)+'%';
  LOCKON.style.width=(b.w*100)+'%'; LOCKON.style.height=(b.h*100)+'%';
  LOCKON.className=''; void LOCKON.offsetWidth;
  LOCKON.className='lockon active';
}}

// ── Mouse events
canvas.addEventListener('mousemove', e => {{
  const r=canvas.getBoundingClientRect();
  const mx=(e.clientX-r.left)*(canvas.width/r.width);
  const my=(e.clientY-r.top)*(canvas.height/r.height);
  document.getElementById('sdcoord').innerHTML='X:'+Math.round(mx)+'&nbsp;&nbsp;Y:'+Math.round(my);
  let found=-1;
  if (scanned) BUILDINGS.forEach((b,i) => {{
    if (mx>=b.x*canvas.width&&mx<=(b.x+b.w)*canvas.width&&
        my>=b.y*canvas.height&&my<=(b.y+b.h)*canvas.height) found=i;
  }});
  if (found!==hovId) {{
    hovId=found; draw();
    if (found>=0) showLockOn(BUILDINGS[found]);
    else LOCKON.className='lockon';
  }}
  if (found>=0) {{
    const b=BUILDINGS[found];
    TIP.style.cssText=`position:absolute;
    display:block;
    left:${{Math.min(e.clientX-r.left+14,r.width-260)}}px;
    top:${{Math.max(e.clientY-r.top-80,4)}}px;
    min-width:220px;
    min-height:80px;
    padding:10px 14px;
    font-size:13px;
    pointer-events:none;z-index:20`;
    document.getElementById('tip-id').textContent='#'+String(found).padStart(4,'0');
    document.getElementById('tip-lbl').style.color=HFILLS[b.label];
    document.getElementById('tip-lbl').textContent=LABELS[b.label];
    document.getElementById('tip-lbl').style.cssText = `margin-top:25px;font-size:16px;`;
    document.getElementById('tip-conf').textContent=Math.round(b.confidence*100)+'%';
  }} else {{ TIP.style.display='none'; }}
}});
canvas.addEventListener('mouseleave',()=>{{
  hovId=-1; TIP.style.display='none'; LOCKON.className='lockon'; draw();
}});
img.addEventListener('load',sync);
if (img.complete) sync();
if (preImg) {{
  preImg.addEventListener('load', () => {{ syncPre(); drawPre(); }});
  if (preImg.complete) {{ syncPre(); drawPre(); }}
}}

// ── Pre-image mouse events
if (preCanvas) {{
  preCanvas.addEventListener('mousemove', e => {{
    const r = preCanvas.getBoundingClientRect();
    const mx = (e.clientX-r.left)*(preCanvas.width/r.width);
    const my = (e.clientY-r.top)*(preCanvas.height/r.height);
    document.getElementById('pre-coord').innerHTML='X:'+Math.round(mx)+'&nbsp;&nbsp;Y:'+Math.round(my);
    let found = -1;
    PRE_BUILDINGS.forEach((b, i) => {{
      if (mx>=b.x*preCanvas.width && mx<=(b.x+b.w)*preCanvas.width &&
          my>=b.y*preCanvas.height && my<=(b.y+b.h)*preCanvas.height) found=i;
    }});
    if (found !== preHovId) {{
      preHovId = found; drawPre();
      if (found >= 0) showPreLockOn(PRE_BUILDINGS[found]);
      else PRE_LOCKON.className = 'lockon';
    }}
    if (found >= 0) {{
      const b = PRE_BUILDINGS[found];
      PRE_TIP.style.cssText = `position:absolute;
      display:block;
      left:${{Math.min(e.clientX-r.left+14,r.width-260)}}px;
      top:${{Math.max(e.clientY-r.top-80,4)}}px;
      min-width:220px;
      min-height:80px;
      padding:10px 14px;
      font-size:13px;
      pointer-events:none;z-index:20`;
      document.getElementById('pre-tip-id').textContent = '#'+String(found).padStart(4,'0');
      document.getElementById('pre-tip-lbl').textContent = 'INTACT';
      document.getElementById('pre-tip-lbl').style.color = '#40c8ff';
      document.getElementById('pre-tip-lbl').style.cssText = `margin-top:25px;font-size:16px;`;
      document.getElementById('pre-tip-status').textContent = 'PRE-DISASTER';
    }} else {{ PRE_TIP.style.display = 'none'; }}
  }});
  preCanvas.addEventListener('mouseleave', () => {{
    preHovId = -1; PRE_TIP.style.display = 'none'; PRE_LOCKON.className = 'lockon'; drawPre();
  }});
}}

// ── GT image: draw immediately on load (no scan animation)
if (gtImg) {{
  gtImg.addEventListener('load', () => {{ syncGt(); drawGt(); }});
  if (gtImg.complete) {{ syncGt(); drawGt(); }}
}}

// ── GT mouse events
if (gtCanvas) {{
  gtCanvas.addEventListener('mousemove', e => {{
    const r = gtCanvas.getBoundingClientRect();
    const mx = (e.clientX-r.left)*(gtCanvas.width/r.width);
    const my = (e.clientY-r.top)*(gtCanvas.height/r.height);
    document.getElementById('gt-coord').innerHTML='X:'+Math.round(mx)+'&nbsp;&nbsp;Y:'+Math.round(my);
    let found = -1;
    GT_BUILDINGS.forEach((b, i) => {{
      if (mx>=b.x*gtCanvas.width && mx<=(b.x+b.w)*gtCanvas.width &&
          my>=b.y*gtCanvas.height && my<=(b.y+b.h)*gtCanvas.height) found=i;
    }});
    if (found !== gtHovId) {{
      gtHovId = found; drawGt();
      if (found >= 0) showGtLockOn(GT_BUILDINGS[found]);
      else GT_LOCKON.className = 'lockon';
    }}
    if (found >= 0) {{
      const b = GT_BUILDINGS[found];
      const lbl = b.label || 0;
      GT_TIP.style.cssText = `position:absolute;
      display:block;
      left:${{Math.min(e.clientX-r.left+14,r.width-260)}}px;
      top:${{Math.max(e.clientY-r.top-80,4)}}px;
      min-width:220px;
      min-height:80px;
      padding:10px 14px;
      font-size:13px;
      pointer-events:none;z-index:20`;
      document.getElementById('gt-tip-id').textContent = '#'+String(found).padStart(4,'0');
      document.getElementById('gt-tip-lbl').style.color = HFILLS[lbl];
      document.getElementById('gt-tip-lbl').textContent = LABELS[lbl];
      document.getElementById('gt-tip-lbl').style.cssText = `margin-top:25px;font-size:16px;`;
      document.getElementById('gt-tip-src').textContent = 'GROUND TRUTH';
    }} else {{ GT_TIP.style.display = 'none'; }}
  }});
  gtCanvas.addEventListener('mouseleave', () => {{
    gtHovId = -1; GT_TIP.style.display = 'none'; GT_LOCKON.className = 'lockon'; drawGt();
  }});
}}
</script>
"""


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

model_choice = st.radio(
    "model",
    [m["key"] for m in MODELS],
    horizontal=True,
    label_visibility="collapsed",
)
selected_model = next(m for m in MODELS if m["key"] == model_choice)

# ── Input source selector
sample_labels = ["— UPLOAD YOUR OWN IMAGES OR SELECT SAMPLES —"] + [s["label"] for s in SAMPLE_PAIRS]
selected_label = st.selectbox(
    "input_source",
    sample_labels,
    label_visibility="collapsed",
)

sample_idx = sample_labels.index(selected_label) - 1  # -1 → upload mode

pre_buildings = []
gt_buildings = []
uploaded_post_json = None
uploaded_pre_json = None

if sample_idx >= 0:
    # ── Sample pair mode
    pair = SAMPLE_PAIRS[sample_idx]
    post_img   = Image.open(pair["post"]).convert("RGB")
    pre_img    = Image.open(pair["pre"]).convert("RGB")
    post_b64   = pil_to_b64(post_img, max_w=600)
    pre_b64    = pil_to_b64(pre_img, max_w=600)
    seed       = pair["seed"]
    event_name = pair["label"]

    # ── Load bundled xBD JSONs as file-like objects for the API
    if pair.get("pre_json") and pair["pre_json"].exists():
        raw_pre = json.loads(pair["pre_json"].read_text())
        if isinstance(raw_pre, dict) and "features" in raw_pre:
            pre_buildings = parse_xbd_json(raw_pre, is_pre=True)
        uploaded_pre_json = io.BytesIO(pair["pre_json"].read_bytes())
    if pair.get("post_json") and pair["post_json"].exists():
        raw_post = json.loads(pair["post_json"].read_text())
        if isinstance(raw_post, dict) and "features" in raw_post:
            gt_buildings = parse_xbd_json(raw_post, is_pre=False)
        uploaded_post_json = io.BytesIO(pair["post_json"].read_bytes())
else:
    # ── Upload mode
    col_pre, col_post = st.columns(2)
    with col_pre:
        uploaded_pre = st.file_uploader(
            "pre", type=["png","jpg","jpeg","tif","tiff"],
            label_visibility="collapsed",
        )
        st.caption("PRE-DISASTER IMAGE (OPTIONAL)")
    with col_post:
        uploaded_post = st.file_uploader(
            "post", type=["png","jpg","jpeg","tif","tiff"],
            label_visibility="collapsed",
        )
        st.caption("POST-DISASTER IMAGE (REQUIRED)")
    col_pre_json, col_post_json = st.columns(2)
    with col_pre_json:
        uploaded_pre_json = st.file_uploader(
            "pre_json", type=["json"],
            label_visibility="collapsed",
        )
        st.caption("PRE-DISASTER JSON (OPTIONAL)")
    with col_post_json:
        uploaded_post_json = st.file_uploader(
            "post_json", type=["json"],
            label_visibility="collapsed",
        )
        st.caption("POST-DISASTER JSON (OPTIONAL)")

    if uploaded_post is None:
        st.markdown(
            "<div style='text-align:center;padding:4rem;color:#0a3050;"
            "font-family:monospace;font-size:12px;letter-spacing:3px;"
            "background:rgba(5,12,20,0.88);border:0.5px solid #0a2a4a;"
            "clip-path:polygon(0 0,calc(100% - 12px) 0,100% 12px,100% 100%,0 100%);'>"
            "[ AWAITING SATELLITE FEED ]<br><br>"
            "<span style='font-size:10px;color:#061828;letter-spacing:2px'>"
            "UPLOAD POST-DISASTER IMAGE TO BEGIN ASSESSMENT</span></div>",
            unsafe_allow_html=True,
        )
        st.stop()

    post_img = Image.open(uploaded_post).convert("RGB")
    post_b64 = pil_to_b64(post_img, max_w=600)
    pre_img  = None
    pre_b64  = None
    if uploaded_pre is not None:
        pre_img = Image.open(uploaded_pre).convert("RGB")
        pre_b64 = pil_to_b64(pre_img, max_w=600)
    seed       = hash(uploaded_post.name) % 9999
    event_name = uploaded_post.name.upper()

    # ── Parse uploaded pre JSON for HUD overlay
    if uploaded_pre_json is not None:
        try:
            raw = json.loads(uploaded_pre_json.getvalue())
            if isinstance(raw, dict) and "features" in raw:
                pre_buildings = parse_xbd_json(raw, is_pre=True)
            elif isinstance(raw, list):
                pre_buildings = [
                    {"x": float(e["x"]), "y": float(e["y"]),
                     "w": float(e["w"]), "h": float(e["h"])}
                    for e in raw
                ]
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            st.error(f"Invalid pre-disaster JSON: {exc}")

    # ── Parse uploaded post JSON for ground truth overlay
    if uploaded_post_json is not None:
        try:
            raw = json.loads(uploaded_post_json.getvalue())
            if isinstance(raw, dict) and "features" in raw:
                gt_buildings = parse_xbd_json(raw, is_pre=False)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            st.error(f"Invalid post-disaster JSON: {exc}")

# ── Prediction
buildings, report = predict_api(
    post_img, pre_img=pre_img,
    post_json_file=uploaded_post_json,
    pre_json_file=uploaded_pre_json,
    seed=seed,
    model_key=selected_model["api"],
)
hud_html = build_hud(pre_b64, post_b64, buildings, pre_buildings, event_name, gt_buildings=gt_buildings, report=report)

st.components.v1.html(hud_html, height=760, scrolling=False)

with st.expander("Raw predictions"):
    st.json({"post_buildings": buildings, "pre_buildings": pre_buildings, "gt_buildings": gt_buildings, "report": report})
