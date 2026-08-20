from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

# Eye landmark connections, same MediaPipe face mesh topology and same reason
# for being copied verbatim rather than imported (see _FACEMESH_LIPS above).
# "left"/"right" as in mediapipe.python.solutions.face_mesh_connections —
# the face's own left/right, not screen-left/right — but since the two eyes
# are otherwise interchangeable here, the labels only need to be consistent,
# not anatomically correct.
_FACEMESH_LEFT_EYE = frozenset(
    [
        (263, 249), (249, 390), (390, 373), (373, 374), (374, 380),
        (380, 381), (381, 382), (382, 362), (263, 466), (466, 388),
        (388, 387), (387, 386), (386, 385), (385, 384), (384, 398),
        (398, 362),
    ]
)  # fmt: skip

_FACEMESH_RIGHT_EYE = frozenset(
    [
        (33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
        (153, 154), (154, 155), (155, 133), (33, 246), (246, 161),
        (161, 160), (160, 159), (159, 158), (158, 157), (157, 173),
        (173, 133),
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

# Google's official model for the MediaPipe Image Segmenter task's "selfie
# segmenter" — a person-vs-background model, used to cut the subject out
# from the background so player.py can animate her without dragging the
# background along with her.
_SEGMENTER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/1/selfie_segmenter.tflite"
)
_SEGMENTER_MODEL_PATH = Path(__file__).parent / "selfie_segmenter.tflite"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        logger.info("Downloading MediaPipe face landmark model (one-time, ~4MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


def _ensure_segmenter_model() -> Path:
    if not _SEGMENTER_MODEL_PATH.exists():
        logger.info("Downloading MediaPipe selfie segmentation model (one-time, ~250KB)...")
        urllib.request.urlretrieve(_SEGMENTER_MODEL_URL, _SEGMENTER_MODEL_PATH)
    return _SEGMENTER_MODEL_PATH


@dataclass(frozen=True)
class RegionGeometry:
    """Pixel-space position/size of a facial region on one specific image, used to drive a warp."""

    center: tuple[float, float]
    width_px: float
    height_px: float


def _detect_landmarks(image: np.ndarray) -> Any:
    """Run MediaPipe's face landmarker on `image` (BGR) and return the first face's landmarks.

    Raises ValueError if no face is detected — callers should fail loudly
    rather than warp a region that doesn't correspond to an actual feature.
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
    return result.face_landmarks[0]


def _region_geometry(landmarks: Any, indices: set[int], w: int, h: int) -> RegionGeometry:
    points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    return RegionGeometry(
        center=((x_min + x_max) / 2, (y_min + y_max) / 2),
        width_px=max(x_max - x_min, 1.0),
        height_px=max(y_max - y_min, 1.0),
    )


def detect_mouth(landmarks: Any, w: int, h: int) -> RegionGeometry:
    """Locate the mouth from already-detected face `landmarks` on a `w`x`h` image."""
    lip_indices = {i for edge in _FACEMESH_LIPS for i in edge}
    return _region_geometry(landmarks, lip_indices, w, h)


def detect_eyes(landmarks: Any, w: int, h: int) -> tuple[RegionGeometry, RegionGeometry]:
    """Locate both eyes from already-detected face `landmarks` on a `w`x`h` image.

    Returns a (left, right) pair — the two are otherwise interchangeable to
    callers, which just need a stable, distinct id per eye.
    """
    left_indices = {i for edge in _FACEMESH_LEFT_EYE for i in edge}
    right_indices = {i for edge in _FACEMESH_RIGHT_EYE for i in edge}
    return (
        _region_geometry(landmarks, left_indices, w, h),
        _region_geometry(landmarks, right_indices, w, h),
    )


def warp_mouth(image: np.ndarray, mouth: RegionGeometry, shape: MouthShape) -> np.ndarray:
    """Return a copy of `image` with the mouth region displaced for one viseme shape.

    Displaces pixels with a Gaussian falloff centered on the mouth, so the
    effect blends smoothly into the rest of the face instead of leaving a
    visible seam — cheap enough to run 9 times (once per viseme) at startup
    and never touch again during playback.
    """
    h, w = image.shape[:2]
    cx, cy = mouth.center
    sigma_x = mouth.width_px * 0.9

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    x_falloff = (xs - cx) ** 2 / (2 * sigma_x**2)

    # Vertical (jaw-open) displacement uses a tall falloff, since it's meant
    # to drag the chin/jaw region down with the lower lip.
    sigma_y_jaw = mouth.height_px * 1.6
    weight_jaw = np.exp(-(x_falloff + (ys - cy) ** 2 / (2 * sigma_y_jaw**2)))

    # Horizontal (pucker/stretch) displacement uses a much shorter falloff,
    # confined to the mouth itself — with the same tall falloff as above this
    # warp reached up into the nose bridge and pinched it into a blade shape.
    sigma_y_mouth = mouth.height_px * 0.6
    weight_mouth = np.exp(-(x_falloff + (ys - cy) ** 2 / (2 * sigma_y_mouth**2)))

    # Only the region at/below the mouth center drops for an open jaw; the
    # upper lip stays roughly put, mimicking how a real jaw hinges open.
    jaw_factor = np.clip((ys - (cy - mouth.height_px * 0.3)) / mouth.height_px, 0.0, 1.0)
    dy_max = mouth.height_px * 1.3
    dx_max = mouth.width_px * 0.35

    disp_y = shape.open_amount * dy_max * weight_jaw * jaw_factor
    disp_x = shape.width * dx_max * weight_mouth * np.sign(xs - cx)

    map_x = (xs - disp_x).astype(np.float32)
    map_y = (ys - disp_y).astype(np.float32)
    return cv2.remap(
        image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def warp_eye_closed(image: np.ndarray, eye: RegionGeometry) -> np.ndarray:
    """Return a BGRA copy of `image` with `eye` compressed shut, faded to transparent
    away from the eye so it can be composited over any mouth-shape frame — for a
    blink — without disturbing the rest of the face.

    Pixels near the eye are pulled vertically toward its center line (an
    eyelid closing), falling off with distance the same way `warp_mouth`
    does; the same falloff becomes the alpha channel so only the localized
    eye region is opaque.
    """
    h, w = image.shape[:2]
    cx, cy = eye.center
    sigma_x = eye.width_px * 0.7
    sigma_y = eye.height_px * 1.8

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    weight = np.exp(-((xs - cx) ** 2 / (2 * sigma_x**2) + (ys - cy) ** 2 / (2 * sigma_y**2)))

    # Never fully closed (closure < 1) so a faint eyelid crease stays visible
    # instead of every pixel in the falloff collapsing onto one dead line.
    closure = 0.85
    map_y = (ys - (ys - cy) * closure * weight).astype(np.float32)
    map_x = xs.astype(np.float32)
    warped = cv2.remap(
        image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    alpha = np.clip(weight * 255, 0, 255).astype(np.uint8)
    return np.dstack([warped, alpha])


def segment_person(image: np.ndarray) -> np.ndarray:
    """Return a `(h, w)` float32 mask of `image` (BGR): 1.0 where a pixel
    belongs to the person, 0.0 for background.

    Uses `output_category_mask` rather than `output_confidence_masks` (which
    would give a nicer soft-edged probability per pixel instead of a hard
    cutoff): the confidence-mask output path segfaults outright with the
    installed mediapipe/model combination, where the category mask works
    reliably. `precompute_frames` softens the resulting hard edge with a
    small blur afterward.
    """
    options = mp_vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_ensure_segmenter_model())),
        output_category_mask=True,
        output_confidence_masks=False,
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with mp_vision.ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(mp_image)
    assert result.category_mask is not None
    # This selfie-segmentation model's category 0 is the person and 255 is
    # background (confirmed empirically against known face vs. background
    # pixels) — the reverse of what "category 0" usually suggests.
    category = result.category_mask.numpy_view().reshape(image.shape[:2])
    return (category == 0).astype(np.float32)


# How far outside her current silhouette the background needs to be
# reconstructed: generous enough to cover the bob, twitch, and
# squash-and-stretch movement player.py applies to her cutout at render
# time, so a small shift never reveals a gap. Doesn't need to be exact —
# everything deeper inside her silhouette is inpainted too, but never seen,
# since her cutout always covers it.
_BACKGROUND_MARGIN_PX = 24


def build_background(image: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    """Reconstruct what's behind the person in `image` (BGR), so player.py has a
    static backdrop to composite her moving cutout over instead of dragging the
    whole photo — background included — along with every animation.

    Inpaints the entire person silhouette (dilated by `_BACKGROUND_MARGIN_PX`)
    from the surrounding background. The result only needs to be convincing in
    that outer margin — deeper inside, it's never actually seen, since her
    cutout always covers it.
    """
    binary_mask = (person_mask > 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_BACKGROUND_MARGIN_PX * 2 + 1, _BACKGROUND_MARGIN_PX * 2 + 1)
    )
    dilated_mask = cv2.dilate(binary_mask, kernel)
    return cv2.inpaint(image, dilated_mask, _BACKGROUND_MARGIN_PX, cv2.INPAINT_TELEA)


@dataclass(frozen=True)
class AvatarAssets:
    """Precomputed rendering assets for one avatar image.

    `background` is a static BGR reconstruction of what's behind her —
    drawn once, never animated. `mouth_frames` is one warped BGRA cutout of
    her (alpha = person segmentation) per Rhubarb viseme shape, meant to be
    composited on top of `background` and animated independently of it.
    `eye_overlays` is one BGRA closed-eye overlay per eye, keyed "left"/
    "right", composited on top of whichever mouth frame is showing to
    animate a blink independently of speech.
    """

    background: np.ndarray
    mouth_frames: dict[str, np.ndarray]
    eye_overlays: dict[str, np.ndarray]


def precompute_frames(image_path: str) -> AvatarAssets:
    """Load `image_path` and precompute its background plate, per-viseme mouth
    cutouts, and eye-blink overlays.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load avatar image: {image_path}")

    h, w = image.shape[:2]
    landmarks = _detect_landmarks(image)
    mouth = detect_mouth(landmarks, w, h)
    left_eye, right_eye = detect_eyes(landmarks, w, h)

    person_mask = segment_person(image)
    background = build_background(image, person_mask)
    # Slightly softened so the cutout's edge blends into `background`
    # instead of showing a hard segmentation seam.
    alpha = np.clip(cv2.GaussianBlur(person_mask, (5, 5), 0) * 255, 0, 255).astype(np.uint8)

    mouth_frames = {
        name: np.dstack([warp_mouth(image, mouth, shape), alpha])
        for name, shape in MOUTH_SHAPES.items()
    }
    eye_overlays = {
        "left": warp_eye_closed(image, left_eye),
        "right": warp_eye_closed(image, right_eye),
    }
    logger.info(
        "Precomputed %d viseme frames and %d eye-blink overlays for avatar image %s",
        len(mouth_frames),
        len(eye_overlays),
        image_path,
    )
    return AvatarAssets(background=background, mouth_frames=mouth_frames, eye_overlays=eye_overlays)
