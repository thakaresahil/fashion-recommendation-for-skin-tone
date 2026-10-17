"""Skin tone estimation from a photo.

Pipeline:
1. Detect the largest frontal face (OpenCV Haar cascade) and crop its central region,
   which avoids hair, background and most of the eyes and mouth.
2. Keep pixels inside a YCrCb skin-colour range, dropping shadows and specular highlights.
3. Convert the remaining pixels to CIELAB and take a trimmed median.
4. Compute the Individual Typology Angle, ITA = arctan((L* - 50) / b*) in degrees, a
   standard dermatology measure of constitutive skin colour (Chardon et al., 1991), and
   map it to one of the tone groups in ``palettes.json``.

Photos are processed in memory and never stored.
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.palettes import PaletteBook, Tone

MAX_SIDE = 800
MIN_SKIN_PIXELS = 400

_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class SkinToneError(ValueError):
    """The image cannot be analysed (unreadable, or no skin found)."""


@dataclass
class SkinToneResult:
    tone: Tone
    ita: float
    lab: tuple[float, float, float]
    skin_hex: str
    face_detected: bool
    face_box: tuple[int, int, int, int] | None
    pixel_count: int
    confidence: float
    alternative: Tone | None = None
    notes: list[str] = field(default_factory=list)


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SkinToneError("Could not decode the image. Please upload a JPEG, PNG or WebP photo.")
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def detect_face(img: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    min_side = max(40, int(min(img.shape[:2]) * 0.12))
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    return (y > 25) & (y < 245) & (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135)


def ita_angle(lightness: float, b: float) -> float:
    return math.degrees(math.atan2(lightness - 50.0, b)) if b > 0 else (90.0 if lightness >= 50 else -90.0)


def analyze(data: bytes, palettes: PaletteBook) -> SkinToneResult:
    img = decode_image(data)
    notes: list[str] = []

    box = detect_face(img)
    if box is not None:
        x, y, w, h = box
        region = img[y + int(0.2 * h) : y + int(0.85 * h), x + int(0.2 * w) : x + int(0.8 * w)]
    else:
        region = img
        notes.append(
            "No face detected, so the whole image was scanned for skin. A clear, front-facing photo works best."
        )

    mask = skin_mask(region)
    pixels = region[mask]
    if box is not None and len(pixels) < MIN_SKIN_PIXELS:
        # The cropped face region is mostly skin anyway; unusual lighting can defeat the colour gate.
        pixels = region.reshape(-1, 3)
        notes.append("Lighting made skin hard to isolate; the estimate may be less accurate.")
    if len(pixels) < MIN_SKIN_PIXELS:
        raise SkinToneError("Not enough visible skin was found. Try a well-lit, front-facing photo.")

    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    # Trim the darkest and brightest 15% of pixels by lightness to suppress shadows and shine.
    lo, hi = np.percentile(lab[:, 0], [15, 85])
    core = lab[(lab[:, 0] >= lo) & (lab[:, 0] <= hi)]
    if len(core) == 0:
        core = lab
    lightness, a, b = (float(v) for v in np.median(core, axis=0))

    ita = ita_angle(lightness, b)
    tone = palettes.from_ita(ita)
    margin = min(abs(ita - tone.ita_min), abs(tone.ita_max - ita))
    alternative = None
    if margin < 4:
        nearer_edge = tone.ita_min if abs(ita - tone.ita_min) < abs(tone.ita_max - ita) else tone.ita_max
        neighbour = palettes.from_ita(nearer_edge - 0.5 if nearer_edge == tone.ita_min else nearer_edge + 0.5)
        alternative = neighbour if neighbour.id != tone.id else None

    confidence = (0.9 if box is not None else 0.55) * min(1.0, len(pixels) / 3000) * (0.6 + 0.4 * min(1.0, margin / 5))

    rgb = cv2.cvtColor(np.array([[[lightness, a, b]]], dtype=np.float32), cv2.COLOR_Lab2RGB)[0, 0]
    skin_hex = "#" + "".join(f"{int(round(np.clip(c, 0, 1) * 255)):02X}" for c in rgb)

    return SkinToneResult(
        tone=tone,
        ita=round(ita, 2),
        lab=(round(lightness, 2), round(a, 2), round(b, 2)),
        skin_hex=skin_hex,
        face_detected=box is not None,
        face_box=box,
        pixel_count=int(len(pixels)),
        confidence=round(confidence, 2),
        alternative=alternative,
        notes=notes,
    )
