"""Skin tone groups and their recommended colour palettes."""

import json
from dataclasses import dataclass
from pathlib import Path

from app.core import colors


@dataclass(frozen=True)
class Tone:
    id: str
    label: str
    swatch: str
    description: str
    colors: tuple[str, ...]
    ita_min: float | None = None
    ita_max: float | None = None


class PaletteBook:
    def __init__(self, tones: list[Tone], common: Tone):
        self.tones = tones
        self.common = common
        self._by_id = {t.id: t for t in [*tones, common]}

    @classmethod
    def load(cls, path: Path) -> "PaletteBook":
        raw = json.loads(path.read_text())

        def build(entry: dict) -> Tone:
            unknown = [c for c in entry["colors"] if colors.canonical(c) is None]
            if unknown:
                raise ValueError(f"Palette '{entry['id']}' uses unknown colours: {unknown}")
            return Tone(
                id=entry["id"],
                label=entry["label"],
                swatch=entry["swatch"],
                description=entry["description"],
                colors=tuple(colors.canonical(c) for c in entry["colors"]),
                ita_min=entry.get("ita_min"),
                ita_max=entry.get("ita_max"),
            )

        return cls([build(t) for t in raw["tones"]], build(raw["common"]))

    def get(self, tone_id: str) -> Tone | None:
        return self._by_id.get(tone_id)

    def all(self) -> list[Tone]:
        return [*self.tones, self.common]

    def from_ita(self, ita: float) -> Tone:
        """Map an Individual Typology Angle to a tone group (clamped at the extremes)."""
        for tone in self.tones:
            if tone.ita_min <= ita < tone.ita_max:
                return tone
        return self.tones[0] if ita >= self.tones[0].ita_min else self.tones[-1]
