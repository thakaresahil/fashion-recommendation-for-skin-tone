"""Colour vocabulary for the catalog: canonical names, swatch hex codes and families.

A *family* groups catalog ``baseColour`` values that should match the same palette
colour, e.g. recommending "Grey" also surfaces "Grey Melange" products.
"""

SWATCHES: dict[str, str] = {
    "Beige": "#D9C7A7",
    "Black": "#1F1F1F",
    "Blue": "#2F6FD6",
    "Bronze": "#9C6B30",
    "Brown": "#7B4A2D",
    "Burgundy": "#7A1F35",
    "Charcoal": "#3B3F45",
    "Coffee Brown": "#5B3A29",
    "Cream": "#F3E9D2",
    "Fluorescent Green": "#5CFF3B",
    "Gold": "#C9A227",
    "Green": "#2E8B57",
    "Grey": "#8C8C8C",
    "Grey Melange": "#A3A3A3",
    "Khaki": "#B5A46E",
    "Lavender": "#B9A6E0",
    "Lime Green": "#8DCB3F",
    "Magenta": "#C2187A",
    "Maroon": "#6E1423",
    "Mauve": "#B784A7",
    "Metallic": "#A8A9AD",
    "Multi": "#888888",
    "Mushroom Brown": "#A08B76",
    "Mustard": "#D1A21E",
    "Navy Blue": "#1F2F5C",
    "Nude": "#E3BC9A",
    "Off White": "#F5F2EA",
    "Olive": "#6B7032",
    "Orange": "#F07A22",
    "Peach": "#F6B99A",
    "Pink": "#EE82A8",
    "Purple": "#6D3FA0",
    "Red": "#C8232C",
    "Rose": "#D9677E",
    "Rust": "#A8481F",
    "Sea Green": "#2E9C80",
    "Silver": "#C0C0C0",
    "Skin": "#E6B79A",
    "Tan": "#C29A6B",
    "Taupe": "#8B7D6B",
    "Teal": "#127C7C",
    "Turquoise Blue": "#30B5C8",
    "White": "#FAFAFA",
    "Yellow": "#F4CD2A",
}

FAMILIES: dict[str, list[str]] = {
    "Grey": ["Grey", "Grey Melange"],
    "Green": ["Green", "Sea Green"],
    "Teal": ["Teal", "Turquoise Blue"],
    "White": ["White", "Off White"],
    "Brown": ["Brown", "Coffee Brown", "Mushroom Brown"],
    "Pink": ["Pink", "Rose"],
    "Yellow": ["Yellow", "Mustard"],
}

# Catalog colour values that are usually other words when they appear in a search query.
_NOT_QUERY_COLOURS = {"Multi", "Skin", "Metallic", "Nude"}

_CANONICAL = {name.lower(): name for name in SWATCHES}


def canonical(name: str) -> str | None:
    """Return the catalog spelling of a colour name (case-insensitive), or None."""
    return _CANONICAL.get(name.strip().lower())


def family(name: str) -> list[str]:
    """All catalog colours that should match ``name``."""
    canon = canonical(name)
    if canon is None:
        return []
    return FAMILIES.get(canon, [canon])


def swatch(name: str) -> str:
    canon = canonical(name)
    return SWATCHES.get(canon, "#888888") if canon else "#888888"


def extract(query: str) -> list[str]:
    """Catalog colours named in free text, longest names first ("navy blue" before "blue")."""
    text = f" {' '.join(query.lower().split())} "
    found = []
    for name in sorted(SWATCHES.keys() - _NOT_QUERY_COLOURS, key=len, reverse=True):
        needle = f" {name.lower()} "
        if needle in text:
            found.append(name)
            text = text.replace(needle, " ")
    return found
