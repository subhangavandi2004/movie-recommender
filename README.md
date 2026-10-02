# Movie Recommender: Recent Popularity vs Collaborative Filtering

Compares four recommenders on 100,000 movie ratings, using a time-based split so every model has to predict the future.

**Data:** MovieTweetings 100K snapshot (ratings from public tweets about films, Feb-Sep 2013, scale 0-10):
16,554 users, 10,506 movies. Source: https://github.com/sidooms/MovieTweetings
(Dooms, De Pessemier and Martens, 2013). Please keep this credit if you publish the repo.

**Run:** `pip install -r requirements.txt`, `python recommender.py`, `pytest test_api.py`, `uvicorn app:app`.
Runs are logged with MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

## Method
- Time-based split: first 60% of ratings train, next 20% validation, last 20% test.
- Settings (SVD size, recent-popularity window) are chosen on validation. The final models are refit on train + validation and scored once on the test period.
- A movie is "liked" when rated 8 or higher. Models rank movies the user has not rated yet; metrics are at K = 10.
- Only users with at least 3 earlier ratings and at least 1 liked movie in the test period are scored (1,956 users).

## Results (test period)

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity (all time) | 0.023 | 0.116 | 0.059 |
| **Recent popularity (last 7 days)** | **0.036** | **0.206** | **0.121** |
| Item-based collaborative filtering | 0.021 | 0.108 | 0.060 |
| SVD matrix factorisation (10 factors) | 0.016 | 0.080 | 0.046 |

## What I learned
Recent popularity is the clear winner: it doubles NDCG@10 over all-time popularity. People tweet about films
they just watched, so what is new matters more than what is generally liked. The personalised models do not beat popularity
because most users have very few ratings (median: 2), which gives collaborative filtering little to learn from.

## Limitations
- Absolute precision is low because many test movies are new releases that did not exist in the training period.
- 7 days is the shortest window I tried (7, 14, 30, 60), so a shorter window might do even better.
- Results are for one time split; the test period covers only a few weeks.

Files: `recommender.py`, `app.py` (FastAPI), `test_api.py`, `results.json`.
