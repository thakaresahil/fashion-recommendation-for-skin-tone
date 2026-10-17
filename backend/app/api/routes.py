from dataclasses import replace
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app import schemas
from app.core import colors
from app.core import skin_tone as tone_detector
from app.core.catalog import Filters
from app.core.palettes import Tone
from app.services import Services

router = APIRouter(prefix="/api/v1")


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def filters_dep(
    gender: Annotated[str | None, Query(description="Men, Women, Boys, Girls or Unisex")] = None,
    category: Annotated[str | None, Query(description="Master category, e.g. Apparel or Footwear")] = None,
    sub_category: Annotated[list[str] | None, Query(description="One or more sub-categories, e.g. Topwear")] = None,
    usage: Annotated[str | None, Query(description="Occasion, e.g. Casual, Formal, Ethnic")] = None,
) -> Filters:
    return Filters(gender=gender, master_category=category, sub_categories=tuple(sub_category or ()), usage=usage)


FiltersDep = Annotated[Filters, Depends(filters_dep)]


def tone_out(tone: Tone) -> schemas.ToneOut:
    has_range = tone.ita_min is not None and tone.ita_max is not None
    return schemas.ToneOut(
        id=tone.id,
        label=tone.label,
        swatch=tone.swatch,
        description=tone.description,
        colors=[schemas.ColorOut(name=c, hex=colors.swatch(c)) for c in tone.colors],
        ita_range=(tone.ita_min, tone.ita_max) if has_range else None,
    )


def product_out(services: Services, row: int) -> schemas.Product:
    r = services.catalog.record(row)
    return schemas.Product(
        id=int(r["id"]),
        name=r["productDisplayName"],
        gender=r["gender"],
        master_category=r["masterCategory"],
        sub_category=r["subCategory"],
        article_type=r["articleType"],
        color=r["baseColour"],
        color_hex=colors.swatch(r["baseColour"]),
        usage=r["usage"],
        year=int(r["year"]),
        image_url=services.image_url(int(r["id"])),
    )


def require_row(services: Services, product_id: int) -> int:
    row = services.catalog.row_of(product_id)
    if row is None:
        raise HTTPException(404, f"Product {product_id} not found")
    return row


@router.get("/health", response_model=schemas.Health, tags=["meta"])
def health(services: ServicesDep):
    return schemas.Health(
        status="ok",
        products=len(services.catalog),
        images=len(services.image_ids),
        search_mode=services.search.mode,
    )


@router.get("/facets", response_model=schemas.Facets, tags=["catalog"])
def facets(services: ServicesDep):
    return services.catalog.facets()


@router.get("/skin-tones", response_model=list[schemas.ToneOut], tags=["skin tone"])
def list_skin_tones(services: ServicesDep):
    return [tone_out(t) for t in services.palettes.all()]


@router.post("/skin-tone/analyze", response_model=schemas.SkinAnalysis, tags=["skin tone"])
async def analyze_skin_tone(services: ServicesDep, image: Annotated[UploadFile, File(description="A face photo")]):
    limit = services.settings.max_upload_mb * 1024 * 1024
    data = await image.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Image is larger than {services.settings.max_upload_mb} MB")
    try:
        result = await run_in_threadpool(tone_detector.analyze, data, services.palettes)
    except tone_detector.SkinToneError as exc:
        raise HTTPException(422, str(exc)) from exc
    lightness, a, b = result.lab
    return schemas.SkinAnalysis(
        tone=tone_out(result.tone),
        alternative=tone_out(result.alternative) if result.alternative else None,
        ita=result.ita,
        lab=schemas.LabColor(l=lightness, a=a, b=b),
        skin_hex=result.skin_hex,
        face_detected=result.face_detected,
        confidence=result.confidence,
        notes=result.notes,
    )


@router.get("/recommendations", response_model=schemas.RecommendationResponse, tags=["recommendations"])
def recommendations(
    services: ServicesDep,
    filters: FiltersDep,
    skin_tone: Annotated[str, Query(description="Tone id from /skin-tones")],
    q: Annotated[str | None, Query(max_length=200, description="Optional free-text refinement")] = None,
    per_color: Annotated[int, Query(ge=1, le=48)] = 8,
    seed: Annotated[int, Query(ge=0, description="Change to shuffle unranked results")] = 0,
):
    tone = services.palettes.get(skin_tone)
    if tone is None:
        raise HTTPException(404, f"Unknown skin tone '{skin_tone}'")
    groups = services.recommender.recommend(tone, filters, q, per_color, seed)
    return schemas.RecommendationResponse(
        tone=tone_out(tone),
        groups=[
            schemas.ColorGroupOut(
                color=schemas.ColorOut(name=g.color, hex=colors.swatch(g.color)),
                total=g.total,
                items=[product_out(services, r) for r in g.rows],
            )
            for g in groups
        ],
    )


@router.get("/products/search", response_model=schemas.SearchResponse, tags=["catalog"])
def search_products(
    services: ServicesDep,
    filters: FiltersDep,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    color: Annotated[list[str] | None, Query(description="Restrict to these base colours")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 24,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    color_names = tuple(name for c in color or () for name in colors.family(c))
    if color and not color_names:
        raise HTTPException(422, f"Unknown colour(s): {color}")
    detected: list[str] = []
    if not color_names:
        # Text retrieval alone lets rare words ("party") outrank the colour a shopper asked for,
        # so colours named in the query become a filter.
        detected = colors.extract(q)
        color_names = tuple(name for c in detected for name in colors.family(c))
    hits, total = services.search.search(
        q, mask=services.catalog.mask(replace(filters, colors=color_names)), limit=limit, offset=offset
    )
    if detected and total == 0:
        detected = []
        hits, total = services.search.search(q, mask=services.catalog.mask(filters), limit=limit, offset=offset)
    return schemas.SearchResponse(
        query=q,
        mode=services.search.mode,
        detected_colors=detected,
        total=total,
        items=[
            schemas.SearchHit(
                product=product_out(services, h.row),
                score=round(h.score, 6),
                sparse_rank=h.sparse_rank,
                dense_rank=h.dense_rank,
            )
            for h in hits
        ],
    )


@router.get("/products/{product_id}", response_model=schemas.Product, tags=["catalog"])
def get_product(services: ServicesDep, product_id: int):
    return product_out(services, require_row(services, product_id))


@router.get("/products/{product_id}/similar", response_model=list[schemas.Product], tags=["catalog"])
def similar_products(
    services: ServicesDep,
    product_id: int,
    limit: Annotated[int, Query(ge=1, le=48)] = 12,
    same_gender: bool = True,
):
    row = require_row(services, product_id)
    record = services.catalog.record(row)
    mask = services.catalog.mask(Filters(gender=record["gender"] if same_gender else None))
    hits = services.search.similar(row, record["text"], mask, limit)
    return [product_out(services, h.row) for h in hits]


@router.get("/products/{product_id}/image", tags=["catalog"], response_class=FileResponse)
def product_image(services: ServicesDep, product_id: int):
    if product_id not in services.image_ids:
        raise HTTPException(404, "No image for this product")
    return FileResponse(services.settings.images_dir / f"{product_id}.jpg", media_type="image/jpeg")
