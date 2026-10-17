import cv2
import numpy as np
import pytest

from app.config import Settings
from app.core.palettes import PaletteBook
from app.core.skin_tone import SkinToneError, analyze, ita_angle

from .conftest import solid_png


@pytest.fixture(scope="module")
def book() -> PaletteBook:
    return PaletteBook.load(Settings().palettes_file)


def test_ita_formula():
    assert ita_angle(50, 20) == pytest.approx(0)
    assert ita_angle(70, 20) == pytest.approx(45)
    assert ita_angle(30, 20) == pytest.approx(-45)


def test_each_tone_swatch_is_classified_as_itself(book):
    for tone in book.tones:
        result = analyze(solid_png(tone.swatch), book)
        assert result.tone.id == tone.id, (tone.id, result.ita)
        assert not result.face_detected
        assert result.notes  # explains that no face was found


def test_lighter_skin_has_higher_ita(book):
    itas = [analyze(solid_png(t.swatch), book).ita for t in book.tones]
    assert itas == sorted(itas, reverse=True)


def test_rejects_non_images(book):
    with pytest.raises(SkinToneError, match="decode"):
        analyze(b"definitely not an image", book)


def test_rejects_images_without_skin(book):
    with pytest.raises(SkinToneError, match="skin"):
        analyze(solid_png("#1E5BFF"), book)


def test_large_images_are_downscaled(book):
    img = np.full((3000, 2000, 3), (130, 160, 210), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    result = analyze(buf.tobytes(), book)
    assert result.pixel_count <= 800 * 800
