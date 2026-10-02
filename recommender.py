"""Movie recommender: popularity vs item-based CF vs SVD matrix factorisation.

Data is split by time (60% train / 20% validation / 20% test) so the models are
always asked to predict the future. Hyper-parameters are picked on validation;
the test set is used once, after refitting on train + validation.
A movie counts as "liked" when the rating is 8 or higher (scale 0-10).
"""
import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

K = 10
LIKE = 8
MIN_HISTORY = 3  # a user needs at least this many earlier ratings to be evaluated


def load():
    r = pd.read_csv("data/ratings.dat", sep="::", engine="python",
                    names=["user", "item", "rating", "ts"])
    return r.sort_values("ts").reset_index(drop=True)


def split(r):
    a, b = int(len(r) * 0.6), int(len(r) * 0.8)
    return r.iloc[:a], r.iloc[a:b], r.iloc[b:]


class Data:
    """Maps user/item ids to matrix rows and columns, built from a training frame."""

    def __init__(self, train):
        self.users = {u: i for i, u in enumerate(train["user"].unique())}
        self.items = {m: i for i, m in enumerate(train["item"].unique())}
        self.item_ids = np.array(list(self.items))
        rows = train["user"].map(self.users).values
        cols = train["item"].map(self.items).values
        shape = (len(self.users), len(self.items))
        self.binary = sparse.csr_matrix((np.ones(len(train), "float32"), (rows, cols)), shape=shape)
        # mean-centred ratings per user, used by the SVD model
        rt = train["rating"].values.astype("float32")
        user_mean = pd.Series(rt).groupby(rows).transform("mean").values
        self.centered = sparse.csr_matrix((rt - user_mean + 1e-3, (rows, cols)), shape=shape)


# each model returns a score matrix function: scores(user_rows) -> dense (n_users, n_items)
def popularity(d):
    pop = np.asarray(d.binary.sum(axis=0)).ravel()
    return lambda rows: np.tile(pop, (len(rows), 1))


def recent_popularity(d, frame, days):
    """Popularity counted only over the last `days` days of the training period."""
    cutoff = frame["ts"].max() - days * 86400
    recent = frame[(frame["ts"] >= cutoff) & frame["item"].isin(d.items)]
    pop = np.zeros(len(d.items))
    for m, c in recent["item"].value_counts().items():
        pop[d.items[m]] = c
    pop = pop + 1e-3 * np.asarray(d.binary.sum(axis=0)).ravel()  # all-time count breaks ties
    return lambda rows: np.tile(pop, (len(rows), 1))


def item_cf(d, shrink=5):
    X = d.binary
    sim = (X.T @ X).astype("float32").tocsr()
    norms = np.sqrt(np.asarray(sim.diagonal()))
    sim = sim.tocoo()
    sim.data = sim.data / (norms[sim.row] * norms[sim.col] + shrink)  # shrunk cosine similarity
    sim = sim.tocsr()
    sim.setdiag(0)
    return lambda rows: (X[rows] @ sim).toarray()


def svd_mf(d, k):
    svd = TruncatedSVD(n_components=k, random_state=42)
    U = svd.fit_transform(d.centered)
    V = svd.components_
    return lambda rows: U[rows] @ V


def evaluate(scores_fn, d, history, target):
    """Precision/Recall/NDCG@K over users with enough history and >=1 liked target item."""
    liked = target[target["rating"] >= LIKE]
    liked = liked[liked["user"].isin(d.users) & liked["item"].isin(d.items)]
    counts = history.groupby("user").size()
    users = [u for u in liked["user"].unique() if counts.get(u, 0) >= MIN_HISTORY]
    if not users:
        return {"n_users": 0}
    rows = np.array([d.users[u] for u in users])
    rel = liked.groupby("user")["item"].apply(set)
    scores = scores_fn(rows).astype("float64")
    scores[d.binary[rows].nonzero()] = -np.inf  # never recommend what the user already rated
    top = np.argsort(-scores, axis=1)[:, :K]
    prec, rec, ndcg = [], [], []
    disc = 1 / np.log2(np.arange(2, K + 2))
    for u, t in zip(users, top):
        recs = d.item_ids[t]
        hit = np.array([m in rel[u] for m in recs], dtype=float)
        prec.append(hit.mean())
        rec.append(hit.sum() / len(rel[u]))
        ideal = disc[: min(K, len(rel[u]))].sum()
        ndcg.append((hit * disc).sum() / ideal)
    return {"n_users": len(users), f"precision@{K}": float(np.mean(prec)),
            f"recall@{K}": float(np.mean(rec)), f"ndcg@{K}": float(np.mean(ndcg))}


def main():
    r = load()
    train, val, test = split(r)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("movie-recommender")

    # 1) choose SVD size on the validation period
    d = Data(train)
    val_scores = {k: evaluate(svd_mf(d, k), d, train, val)[f"ndcg@{K}"] for k in (10, 20, 50)}
    best_k = max(val_scores, key=val_scores.get)
    win_scores = {w: evaluate(recent_popularity(d, train, w), d, train, val)[f"ndcg@{K}"]
                  for w in (7, 14, 30, 60)}
    best_w = max(win_scores, key=win_scores.get)

    # 2) refit on train + validation and evaluate once on the test period
    full = pd.concat([train, val])
    d = Data(full)
    models = {"popularity": popularity(d),
              f"recent_popularity_{best_w}d": recent_popularity(d, full, best_w),
              "item_based_cf": item_cf(d),
              f"svd_{best_k}_factors": svd_mf(d, best_k)}
    results = {"ratings": len(r), "users": int(r["user"].nunique()), "items": int(r["item"].nunique()),
               "svd_validation_ndcg": {str(k): round(v, 4) for k, v in val_scores.items()},
               "chosen_svd_factors": best_k,
               "recent_popularity_validation_ndcg": {str(k): round(v, 4) for k, v in win_scores.items()},
               "chosen_recent_window_days": best_w, "test": {}}
    for name, fn in models.items():
        m = evaluate(fn, d, full, test)
        results["test"][name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in m.items() if k != "n_users"})

    # artefacts for the API: best model = recent popularity, minus what each user already rated
    best = max(results["test"], key=lambda n: results["test"][n][f"ndcg@{K}"])
    results["best_model_on_test"] = best
    cutoff = full["ts"].max() - best_w * 86400
    recent = full[full["ts"] >= cutoff]["item"].value_counts()
    movies = pd.read_csv("data/movies.dat", sep="::", engine="python", names=["item", "title", "genres"],
                         encoding="latin-1")
    titles = dict(zip(movies["item"], movies["title"]))
    popular = [int(i) for i in recent.index[:200]]
    seen = full.groupby("user")["item"].apply(lambda x: [int(i) for i in x]).to_dict()
    joblib.dump({"popular": popular, "seen": {int(u): v for u, v in seen.items()},
                 "titles": {int(i): str(t) for i, t in titles.items()}, "window_days": best_w},
                "recs.joblib")
    json.dump(results, open("results.json", "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
