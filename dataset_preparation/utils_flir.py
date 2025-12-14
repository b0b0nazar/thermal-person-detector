# dataset_preparation/utils_flir.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def coco_bbox_to_yolo(bbox, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    # bbox is [x_min, y_min, width, height] (COCO style)
    x, y, w, h = bbox
    xc = (x + w / 2.0) / float(img_w)
    yc = (y + h / 2.0) / float(img_h)
    wn = w / float(img_w)
    hn = h / float(img_h)
    return xc, yc, wn, hn

def clamp01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
