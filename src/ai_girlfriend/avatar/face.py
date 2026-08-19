from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from ai_girlfriend.avatar.visemes import MOUTH_SHAPES, MouthShape

logger = logging.getLogger(__name__)

# The lip landmark connections from MediaPipe's face mesh (outer + inner lip
# contour, 40 edges over indices into the 478-point face landmark set).
# Copied verbatim from mediapipe.python.solutions.face_mesh_connections
# .FACEMESH_LIPS rather than imported: that legacy `solutions` module was
# removed from the pip package as of MediaPipe 1.0 in favor of the Tasks API
# (`mediapipe.tasks.python.vision.FaceLandmarker`, used below), which has no
# equivalent named constant of its own — see
# https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
_FACEMESH_LIPS = frozenset(
    [
        (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
        (17, 314), (314, 405), (405, 321), (321, 375),
        (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
        (37, 0), (0, 267),
        (267, 269), (269, 270), (270, 409), (409, 291),
        (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
        (14, 317), (317, 402), (402, 318), (318, 324),
        (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
        (82, 13), (13, 312), (312, 311), (311, 310),
        (310, 415), (415, 308),
    ]
)  # fmt: skip

# Google's official model bundle for the MediaPipe Face Landmarker task
# (https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).
# Downloaded once and cached alongside this module — same one-time-download
# pattern faster-whisper already uses for STT models.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent / "face_landmarker.task"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        logger.info("Downloading MediaPipe face landmark model (one-time, ~4MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


@dataclass(frozen=True)
class MouthGeometry:
    """Pixel-space mouth position/size on one specific image, used to drive the warp."""

    center: tuple[float, float]
    width_px: float
    height_px: float


def detect_mouth(image: np.ndarray) -> MouthGeometry:
    """Locate the mouth in `image` (BGR, as returned by cv2.imread).

    Raises ValueError if no face is detected — callers should fail loudly
    rather than warp a region that doesn't correspond to an actual mouth.
    """
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_ensure_model())),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise ValueError("No face detected in the avatar image")

    h, w = image.shape[:2]
    lip_indices = {i for edge in _FACEMESH_LIPS for i in edge}
    landmarks = result.face_landmarks[0]
    points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in lip_indices])

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    return MouthGeometry(
        center=((x_min + x_max) / 2, (y_min + y_max) / 2),
        width_px=max(x_max - x_min, 1.0),
        height_px=max(y_max - y_min, 1.0),
    )


def warp_mouth(image: np.ndarray, mouth: MouthGeometry, shape: MouthShape) -> np.ndarray:
    """Return a copy of `image` with the mouth region displaced for one viseme shape.

    Displaces pixels with a Gaussian falloff centered on the mouth, so the
    effect blends smoothly into the rest of the face instead of leaving a
    visible seam — cheap enough to run 9 times (once per viseme) at startup
    and never touch again during playback.
    """
    h, w = image.shape[:2]
    cx, cy = mouth.center
    sigma_x = mouth.width_px * 0.9
    sigma_y = mouth.height_px * 1.6

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    weight = np.exp(-(((xs - cx) ** 2) / (2 * sigma_x**2) + ((ys - cy) ** 2) / (2 * sigma_y**2)))

    # Only the region at/below the mouth center drops for an open jaw; the
    # upper lip stays roughly put, mimicking how a real jaw hinges open.
    jaw_factor = np.clip((ys - (cy - mouth.height_px * 0.3)) / mouth.height_px, 0.0, 1.0)
    dy_max = mouth.height_px * 1.3
    dx_max = mouth.width_px * 0.5

    disp_y = shape.open_amount * dy_max * weight * jaw_factor
    disp_x = shape.width * dx_max * weight * np.sign(xs - cx)

    map_x = (xs - disp_x).astype(np.float32)
    map_y = (ys - disp_y).astype(np.float32)
    return cv2.remap(
        image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def precompute_frames(image_path: str) -> dict[str, np.ndarray]:
    """Load `image_path` and precompute one warped BGR frame per Rhubarb viseme shape."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load avatar image: {image_path}")

    mouth = detect_mouth(image)
    frames = {name: warp_mouth(image, mouth, shape) for name, shape in MOUTH_SHAPES.items()}
    logger.info("Precomputed %d viseme frames for avatar image %s", len(frames), image_path)
    return frames
