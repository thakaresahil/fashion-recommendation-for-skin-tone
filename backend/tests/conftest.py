import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    empty = tmp_path_factory.mktemp("empty")
    return Settings(enable_dense=False, images_dir=empty / "images", embeddings_dir=empty / "embeddings")


@pytest.fixture(scope="session")
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture(scope="session")
def services(client):
    return client.app.state.services


def solid_png(hex_color: str, size: int = 200) -> bytes:
    rgb = [int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
    ok, buf = cv2.imencode(".png", np.full((size, size, 3), rgb[::-1], dtype=np.uint8))
    assert ok
    return buf.tobytes()
