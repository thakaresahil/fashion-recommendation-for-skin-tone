"""Public API response models."""

from pydantic import BaseModel, Field


class ColorOut(BaseModel):
    name: str
    hex: str


class ToneOut(BaseModel):
    id: str
    label: str
    swatch: str
    description: str
    colors: list[ColorOut]
    ita_range: tuple[float, float] | None = None


class Product(BaseModel):
    id: int
    name: str
    gender: str
    master_category: str
    sub_category: str
    article_type: str
    color: str
    color_hex: str
    usage: str
    year: int
    image_url: str | None = None


class ColorGroupOut(BaseModel):
    color: ColorOut
    total: int = Field(description="Products in this colour that match the filters")
    items: list[Product]


class RecommendationResponse(BaseModel):
    tone: ToneOut
    groups: list[ColorGroupOut]


class LabColor(BaseModel):
    l: float
    a: float
    b: float


class SkinAnalysis(BaseModel):
    tone: ToneOut
    alternative: ToneOut | None = Field(None, description="Neighbouring tone when the estimate is near a boundary")
    ita: float = Field(description="Individual Typology Angle in degrees")
    lab: LabColor
    skin_hex: str
    face_detected: bool
    confidence: float = Field(ge=0, le=1)
    notes: list[str]


class SearchHit(BaseModel):
    product: Product
    score: float
    sparse_rank: int | None
    dense_rank: int | None


class SearchResponse(BaseModel):
    query: str
    mode: str
    detected_colors: list[str] = Field(description="Colours parsed from the query and applied as a filter")
    total: int
    items: list[SearchHit]


class FacetValue(BaseModel):
    value: str
    count: int


class Facets(BaseModel):
    gender: list[FacetValue]
    masterCategory: list[FacetValue]
    subCategory: list[FacetValue]
    usage: list[FacetValue]
    baseColour: list[FacetValue]


class Health(BaseModel):
    status: str
    products: int
    images: int
    search_mode: str
