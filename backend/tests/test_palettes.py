import pytest

from app.config import Settings
from app.core import colors
from app.core.palettes import PaletteBook


@pytest.fixture(scope="module")
def book() -> PaletteBook:
    return PaletteBook.load(Settings().palettes_file)


def test_bands_are_contiguous(book):
    for upper, lower in zip(book.tones, book.tones[1:], strict=False):
        assert upper.ita_min == lower.ita_max


@pytest.mark.parametrize(
    ("ita", "expected"),
    [
        (80, "very_fair"),
        (55, "very_fair"),
        (54.9, "fair"),
        (30, "olive"),
        (0, "brown"),
        (-45, "deep"),
        (120, "very_fair"),
        (-120, "deep"),
    ],
)
def test_from_ita(book, ita, expected):
    assert book.from_ita(ita).id == expected


def test_palette_colours_exist_in_catalog(book, services):
    catalog_colours = set(services.catalog.df["baseColour"])
    for tone in book.all():
        for colour in tone.colors:
            assert set(colors.family(colour)) & catalog_colours, f"{tone.id}: no products in {colour}"


def test_colour_lookup_is_case_insensitive():
    assert colors.canonical("navy blue") == "Navy Blue"
    assert colors.family("grey") == ["Grey", "Grey Melange"]
    assert colors.family("not-a-colour") == []
