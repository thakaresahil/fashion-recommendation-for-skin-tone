import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from scripts import fetch_images

from .conftest import solid_png


def jpeg(color=(40, 80, 200)) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((80, 60, 3), color, dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_sample_is_deterministic_and_covers_every_colour(services):
    df = services.catalog.df
    a = fetch_images.sample_ids(df, 1500)
    assert a == fetch_images.sample_ids(df, 1500)
    assert 1000 < len(a) < 3500
    sampled = df[df["id"].isin(a)]
    assert set(sampled["baseColour"]) == set(df["baseColour"])


def test_local_folder_and_zip_sources(tmp_path):
    folder = tmp_path / "archive" / "images"
    folder.mkdir(parents=True)
    for pid in (1, 2, 3):
        (folder / f"{pid}.jpg").write_bytes(jpeg())
    (folder / "styles.csv").write_text("id\n1\n")
    assert sorted(pid for pid, _ in fetch_images.from_local(tmp_path / "archive", {1, 3, 99})) == [1, 3]

    zipped = tmp_path / "archive.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.writestr("images/7.jpg", jpeg())
        zf.writestr("images/8.jpg", jpeg())
        zf.writestr("styles.csv", "id\n7\n")
    assert [pid for pid, _ in fetch_images.from_local(zipped, {8})] == [8]


def test_reads_hugging_face_parquet_layout(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(
        {
            "id": [10, 11, 12],
            "productDisplayName": ["a", "b", "c"],
            "image": [{"bytes": jpeg(), "path": None}, {"bytes": jpeg(), "path": None}, None],
        }
    )
    pq.write_table(table, tmp_path / "train.parquet")
    rows = list(fetch_images.read_parquet(pq.ParquetFile(tmp_path / "train.parquet"), {11, 12}))
    assert [pid for pid, _ in rows] == [11]


def test_fetch_writes_jpegs_and_converts_other_formats(tmp_path):
    wanted = {1, 2}
    written = fetch_images.fetch(iter([(1, jpeg()), (2, solid_png("#336699"))]), tmp_path, wanted)
    assert written == 2 and wanted == set()
    for pid in (1, 2):
        assert (tmp_path / f"{pid}.jpg").read_bytes()[:2] == fetch_images.JPEG_MAGIC


@pytest.fixture(scope="module")
def client_with_images(tmp_path_factory, services):
    images = tmp_path_factory.mktemp("images")
    df = services.catalog.df
    # One imaged product per colour of the "brown" palette, for women.
    ids = []
    for colour in ["Green", "Navy Blue", "Blue", "Yellow", "White"]:
        pid = int(df[(df["gender"] == "Women") & (df["baseColour"] == colour)]["id"].iloc[5])
        (images / f"{pid}.jpg").write_bytes(jpeg())
        ids.append(pid)
    settings = Settings(enable_dense=False, images_dir=images, embeddings_dir=images / "none")
    with TestClient(create_app(settings)) as c:
        yield c, ids


def test_recommendations_put_products_with_images_first(client_with_images):
    client, ids = client_with_images
    body = client.get("/api/v1/recommendations", params={"skin_tone": "brown", "gender": "Women"}).json()
    for group in body["groups"]:
        first = group["items"][0]
        assert first["id"] in ids
        assert first["image_url"] == f"/api/v1/products/{first['id']}/image"
        assert all(item["image_url"] is None for item in group["items"][1:])

    res = client.get(f"/api/v1/products/{ids[0]}/image")
    assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg"
    assert client.get("/api/v1/health").json()["images"] == 5


def test_image_base_url_setting(tmp_path):
    settings = Settings(enable_dense=False, images_dir=tmp_path, image_base_url="https://cdn.example.com/img/")
    with TestClient(create_app(settings)) as c:
        item = c.get("/api/v1/products/search", params={"q": "shirt", "limit": 1}).json()["items"][0]["product"]
    assert item["image_url"] == f"https://cdn.example.com/img/{item['id']}.jpg"


def test_image_id_parsing():
    assert fetch_images._image_id("images/123.jpg") == 123
    assert fetch_images._image_id("images/123.JPG") == 123
    assert fetch_images._image_id("styles.csv") is None
    assert fetch_images._image_id("__MACOSX/._123.jpg") is None
