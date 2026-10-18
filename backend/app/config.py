"""Application settings, loaded from environment variables (prefix ``HUEMATCH_``)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HUEMATCH_", env_file=".env", extra="ignore")

    app_name: str = "HueMatch API"
    data_dir: Path = REPO_ROOT / "data"
    catalog_files: list[str] = ["men.csv", "women.csv"]
    images_dir: Path = REPO_ROOT / "data" / "images"
    # Serve images from a CDN or bucket instead, as {image_base_url}/{id}.jpg for every product.
    image_base_url: str | None = None
    embeddings_dir: Path = REPO_ROOT / "data" / "embeddings"
    palettes_file: Path = Path(__file__).resolve().parent / "data" / "palettes.json"

    # Weight of the dense (CLIP) retriever in rank fusion; BM25 gets 1 - dense_weight.
    dense_weight: float = 0.6
    clip_model: str = "sentence-transformers/clip-ViT-B-32"
    enable_dense: bool = True

    max_upload_mb: int = 8
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
