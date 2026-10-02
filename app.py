"""FastAPI service for the recommender. Run: uvicorn app:app --reload
Serves the best model from the evaluation (recent popularity), hiding movies the user already rated.
"""
import joblib
from fastapi import FastAPI, Query

art = joblib.load("recs.joblib")
app = FastAPI(title="Movie Recommender API")


@app.get("/health")
def health():
    return {"status": "ok", "recent_window_days": art["window_days"]}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = Query(10, ge=1, le=50)):
    seen = set(art["seen"].get(user_id, []))
    items = [m for m in art["popular"] if m not in seen][:k]
    return {"user_id": user_id, "known_user": user_id in art["seen"],
            "recommendations": [{"movie_id": m, "title": art["titles"].get(m, "unknown")} for m in items]}
