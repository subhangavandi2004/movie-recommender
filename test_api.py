from fastapi.testclient import TestClient

from app import app, art

client = TestClient(app)


def test_known_user_gets_unseen_movies():
    uid = next(u for u, s in art["seen"].items() if len(s) >= 3)
    recs = client.get(f"/recommend/{uid}?k=10").json()["recommendations"]
    assert len(recs) == 10
    assert not {r["movie_id"] for r in recs} & set(art["seen"][uid])


def test_new_user_gets_popular_list():
    r = client.get("/recommend/999999999?k=5").json()
    assert r["known_user"] is False and len(r["recommendations"]) == 5


def test_bad_k_rejected():
    assert client.get("/recommend/1?k=0").status_code == 422
