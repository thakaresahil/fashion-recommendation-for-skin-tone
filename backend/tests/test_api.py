from .conftest import solid_png


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["products"] > 20000
    assert body["search_mode"] == "bm25"


def test_skin_tones(client):
    tones = client.get("/api/v1/skin-tones").json()
    assert [t["id"] for t in tones][-1] == "common"
    assert all(c["hex"].startswith("#") for t in tones for c in t["colors"])


def test_recommendations_are_in_palette_colours(client):
    res = client.get("/api/v1/recommendations", params={"skin_tone": "olive", "gender": "Men", "per_color": 5})
    assert res.status_code == 200
    body = res.json()
    assert [g["color"]["name"] for g in body["groups"]] == [c["name"] for c in body["tone"]["colors"]]
    for group in body["groups"]:
        assert len(group["items"]) <= 5
        for item in group["items"]:
            assert item["gender"] in {"Men", "Unisex"}
            assert item["color"] in {group["color"]["name"]} | {"Grey Melange", "Sea Green", "Off White"}


def test_recommendations_seed_shuffles_and_query_ranks(client):
    base = {"skin_tone": "brown", "gender": "Women", "per_color": 6}
    a = client.get("/api/v1/recommendations", params={**base, "seed": 1}).json()
    b = client.get("/api/v1/recommendations", params={**base, "seed": 2}).json()
    assert a != b
    ranked = client.get("/api/v1/recommendations", params={**base, "q": "kurta"}).json()
    assert any("kurta" in i["name"].lower() for g in ranked["groups"] for i in g["items"])


def test_recommendations_unknown_tone(client):
    assert client.get("/api/v1/recommendations", params={"skin_tone": "nope"}).status_code == 404


def test_search(client):
    body = client.get("/api/v1/products/search", params={"q": "formal shirt", "color": "navy blue"}).json()
    assert body["total"] > 0
    assert all(h["product"]["color"] == "Navy Blue" for h in body["items"])
    assert client.get("/api/v1/products/search", params={"q": "shirt", "color": "plaid"}).status_code == 422


def test_search_applies_colours_named_in_query(client):
    body = client.get("/api/v1/products/search", params={"q": "red party dress", "limit": 10}).json()
    assert body["detected_colors"] == ["Red"]
    assert all(h["product"]["color"] == "Red" for h in body["items"])
    assert all(h["product"]["article_type"] == "Dresses" for h in body["items"][:5])
    # A colour with no matching products falls back to unfiltered results.
    body = client.get(
        "/api/v1/products/search", params={"q": "taupe sports shoes", "category": "Footwear", "gender": "Men"}
    ).json()
    assert body["detected_colors"] == [] and body["total"] > 0


def test_product_detail_and_similar(client):
    first = client.get("/api/v1/products/search", params={"q": "sports shoes"}).json()["items"][0]["product"]
    assert client.get(f"/api/v1/products/{first['id']}").json() == first
    similar = client.get(f"/api/v1/products/{first['id']}/similar", params={"limit": 4}).json()
    assert len(similar) == 4 and first["id"] not in [p["id"] for p in similar]
    assert client.get("/api/v1/products/999999999").status_code == 404
    assert client.get(f"/api/v1/products/{first['id']}/image").status_code == 404


def test_analyze_endpoint(client):
    res = client.post("/api/v1/skin-tone/analyze", files={"image": ("face.png", solid_png("#976E56"), "image/png")})
    assert res.status_code == 200
    body = res.json()
    assert body["tone"]["id"] == "brown"
    assert 0 <= body["confidence"] <= 1
    bad = client.post("/api/v1/skin-tone/analyze", files={"image": ("x.txt", b"hello", "text/plain")})
    assert bad.status_code == 422


def test_facets(client):
    facets = client.get("/api/v1/facets").json()
    assert {"Men", "Women"} <= {f["value"] for f in facets["gender"]}
